def BasalGanglia(dimensions, n_neurons_per_ensemble=100, output_weight=-3.,
                 input_bias=0., ampa_config=None, gaba_config=None, net=None):
    """Winner take all network, typically used for action selection.

    The basal ganglia network outputs approximately 0 at the dimension with
    the largest value, and is negative elsewhere.

    While the basal ganglia is primarily defined by its winner-take-all
    function, it is also organized to match the organization of the human
    basal ganglia. It consists of five ensembles:

    * Striatal D1 dopamine-receptor neurons (``strD1``)
    * Striatal D2 dopamine-receptor neurons (``strD2``)
    * Subthalamic nucleus (``stn``)
    * Globus pallidus internus / substantia nigra reticulata (``gpi``)
    * Globus pallidus externus (``gpe``)

    Interconnections between these areas are also based on known
    neuroanatomical connections. See [1]_ for more details, and [2]_ for
    the original non-spiking basal ganglia model by
    Gurney, Prescott & Redgrave that this model is based on.

    .. note:: The default `.Solver` for the basal ganglia is `.NnlsL2nz`, which
              requires SciPy. If SciPy is not installed, the global default
              solver will be used instead.

    Parameters
    ----------
    dimensions : int
        Number of dimensions (i.e., actions).
    n_neurons_per_ensemble : int, optional (Default: 100)
        Number of neurons in each ensemble in the network.
    output_weight : float, optional (Default: -3.)
        A scaling factor on the output of the basal ganglia
        (specifically on the connection out of the GPi).
    input_bias : float, optional (Default: 0.)
        An amount by which to bias all dimensions of the input node.
        Biasing the input node is important for ensuring that all input
        dimensions are positive and easily comparable.
    ampa_config : config, optional (Default: None)
        Configuration for connections corresponding to biological connections
        to AMPA receptors (i.e., connections from STN to to GPi and GPe).
        If None, a default configuration using a 2 ms lowpass synapse
        will be used.
    gaba_config : config, optional (Default: None)
        Configuration for connections corresponding to biological connections
        to GABA receptors (i.e., connections from StrD1 to GPi, StrD2 to GPe,
        and GPe to GPi and STN). If None, a default configuration using an
        8 ms lowpass synapse will be used.
    net : Network, optional (Default: None)
        A network in which the network components will be built.
        This is typically used to provide a custom set of Nengo object
        defaults through modifying ``net.config``.

    Returns
    -------
    net : Network
        The newly built basal ganglia network, or the provided ``net``.

    Attributes
    ----------
    net.bias_input : Node or None
        If ``input_bias`` is non-zero, this node will be created to bias
        all of the dimensions of the input signal.
    net.gpe : EnsembleArray
        Globus pallidus externus ensembles.
    net.gpi : EnsembleArray
        Globus pallidus internus ensembles.
    net.input : Node
        Accepts the input signal.
    net.output : Node
        Provides the output signal.
    net.stn : EnsembleArray
        Subthalamic nucleus ensembles.
    net.strD1 : EnsembleArray
        Striatal D1 ensembles.
    net.strD2 : EnsembleArray
        Striatal D2 ensembles.

    References
    ----------
    .. [1] Stewart, T. C., Choo, X., & Eliasmith, C. (2010).
       Dynamic behaviour of a spiking model of action selection in the
       basal ganglia. In Proceedings of the 10th international conference on
       cognitive modeling (pp. 235-40).
    .. [2] Gurney, K., Prescott, T., & Redgrave, P. (2001).
       A computational model of action selection in the basal
       ganglia. Biological Cybernetics 84, 401-423.
    """

    if net is None:
        net = nengo.Network("Basal Ganglia")

    ampa_config, override_ampa = config_with_default_synapse(
        ampa_config, nengo.Lowpass(0.002))
    gaba_config, override_gaba = config_with_default_synapse(
        gaba_config, nengo.Lowpass(0.008))

    # Affects all ensembles / connections in the BG
    # unless they've been overridden on `net.config`
    config = nengo.Config(nengo.Ensemble, nengo.Connection)
    config[nengo.Ensemble].radius = 1.5
    config[nengo.Ensemble].encoders = Choice([[1]])
    try:
        # Best, if we have SciPy
        config[nengo.Connection].solver = NnlsL2nz()
    except ImportError:
        # Warn if we can't use the better decoder solver.
        warnings.warn("SciPy is not installed, so BasalGanglia will "
                      "use the default decoder solver. Installing SciPy "
                      "may improve BasalGanglia performance.")

    ea_params = {'n_neurons': n_neurons_per_ensemble,
                 'n_ensembles': dimensions}

    with config, net:
        net.strD1 = EnsembleArray(label="Striatal D1 neurons",
                                  intercepts=Uniform(Weights.e, 1),
                                  **ea_params)
        net.strD2 = EnsembleArray(label="Striatal D2 neurons",
                                  intercepts=Uniform(Weights.e, 1),
                                  **ea_params)
        net.stn = EnsembleArray(label="Subthalamic nucleus",
                                intercepts=Uniform(Weights.ep, 1),
                                **ea_params)
        net.gpi = EnsembleArray(label="Globus pallidus internus",
                                intercepts=Uniform(Weights.eg, 1),
                                **ea_params)
        net.gpe = EnsembleArray(label="Globus pallidus externus",
                                intercepts=Uniform(Weights.ee, 1),
                                **ea_params)

        net.input = nengo.Node(label="input", size_in=dimensions)
        net.output = nengo.Node(label="output", size_in=dimensions)

        # add bias input (BG performs best in the range 0.5--1.5)
        if abs(input_bias) > 0.0:
            net.bias_input = nengo.Node(np.ones(dimensions) * input_bias,
                                        label="basal ganglia bias")
            nengo.Connection(net.bias_input, net.input)

        # spread the input to StrD1, StrD2, and STN
        nengo.Connection(net.input, net.strD1.input, synapse=None,
                         transform=Weights.ws * (1 + Weights.lg))
        nengo.Connection(net.input, net.strD2.input, synapse=None,
                         transform=Weights.ws * (1 - Weights.le))
        nengo.Connection(net.input, net.stn.input, synapse=None,
                         transform=Weights.wt)

        # connect the striatum to the GPi and GPe (inhibitory)
        strD1_output = net.strD1.add_output('func_str', Weights.str_func)
        strD2_output = net.strD2.add_output('func_str', Weights.str_func)
        with gaba_config:
            nengo.Connection(strD1_output, net.gpi.input,
                             transform=-Weights.wm)
            nengo.Connection(strD2_output, net.gpe.input,
                             transform=-Weights.wm)

        # connect the STN to GPi and GPe (broad and excitatory)
        tr = Weights.wp * np.ones((dimensions, dimensions))
        stn_output = net.stn.add_output('func_stn', Weights.stn_func)
        with ampa_config:
            nengo.Connection(stn_output, net.gpi.input, transform=tr)
            nengo.Connection(stn_output, net.gpe.input, transform=tr)

        # connect the GPe to GPi and STN (inhibitory)
        gpe_output = net.gpe.add_output('func_gpe', Weights.gpe_func)
        with gaba_config:
            nengo.Connection(gpe_output, net.gpi.input, transform=-Weights.we)
            nengo.Connection(gpe_output, net.stn.input, transform=-Weights.wg)

        # connect GPi to output (inhibitory)
        gpi_output = net.gpi.add_output('func_gpi', Weights.gpi_func)
        nengo.Connection(gpi_output, net.output, synapse=None,
                         transform=output_weight)

    # Return ampa_config and gaba_config to previous states, if changed
    if override_ampa:
        del ampa_config[nengo.Connection].synapse
    if override_gaba:
        del gaba_config[nengo.Connection].synapse

    return net