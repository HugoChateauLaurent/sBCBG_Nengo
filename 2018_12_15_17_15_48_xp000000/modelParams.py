#!/apps/free/python/2.7.10/bin/python 

## This file was auto-generated by run.py called with the following arguments:
# run.py --platform Local --whichTest sBCBG --nbcpu -1 --custom baseParams.py --simulator Nengo

## ID string of experiment:
# 2018_12_15_17_15_48_xp000000

## Reproducibility info:
#  platform = Local
#  git commit ID = d63ee0c4ed50e61e56a895816544b90dac18ff2b
#  Changes not yet commited in the following files:  M baseParams.py -  M baseParams.pyc -  M miscellaneous/.ipynb_checkpoints/nengo_nest-checkpoint.ipynb -  M miscellaneous/nengo_nest.ipynb -  M sBCBG.py - 

params =\
{
    "GArkyArky": 0.2,
    "GArkyFSI": 1.0,
    "GArkyMSN": 1.0,
    "GArkyProt": 0.2,
    "GCMPfArky": 1.0,
    "GCMPfFSI": 1.0,
    "GCMPfGPe": 1.0,
    "GCMPfGPi": 1.0,
    "GCMPfMSN": 1.0,
    "GCMPfProt": 1.0,
    "GCMPfSTN": 1.0,
    "GCSNFSI": 1.0,
    "GCSNMSN": 1.0,
    "GFSIFSI": 1.0,
    "GFSIMSN": 1.0,
    "GGPeFSI": 1.0,
    "GGPeGPe": 1.0,
    "GGPeGPi": 1.0,
    "GGPeMSN": 1.0,
    "GGPeSTN": 1.0,
    "GMSNArky": 1.0,
    "GMSNGPe": 1.0,
    "GMSNGPi": 1.0,
    "GMSNMSN": 1.0,
    "GMSNProt": 1.0,
    "GPTNFSI": 1.0,
    "GPTNMSN": 1.0,
    "GPTNSTN": 1.0,
    "GProtArky": 0.8,
    "GProtGPi": 1.0,
    "GProtProt": 0.8,
    "GProtSTN": 1.0,
    "GSTNArky": 1.0,
    "GSTNFSI": 1.0,
    "GSTNGPe": 1.0,
    "GSTNGPi": 1.0,
    "GSTNMSN": 1.0,
    "GSTNProt": 1.0,
    "IeArky": -50.0,
    "IeFSI": -50.0,
    "IeGPe": -50.0,
    "IeGPi": -50.0,
    "IeMSN": -50.0,
    "IeProt": -50.0,
    "IeSTN": -50.0,
    "LG14modelID": 9,
    "RedundancyType": "outDegreeAbs",
    "cTypeArkyArky": "diffuse",
    "cTypeArkyFSI": "diffuse",
    "cTypeArkyMSN": "diffuse",
    "cTypeArkyProt": "diffuse",
    "cTypeCMPfArky": "diffuse",
    "cTypeCMPfFSI": "diffuse",
    "cTypeCMPfGPe": "diffuse",
    "cTypeCMPfGPi": "diffuse",
    "cTypeCMPfMSN": "diffuse",
    "cTypeCMPfProt": "diffuse",
    "cTypeCMPfSTN": "diffuse",
    "cTypeCSNFSI": "focused",
    "cTypeCSNMSN": "focused",
    "cTypeFSIFSI": "diffuse",
    "cTypeFSIMSN": "diffuse",
    "cTypeGPeFSI": "diffuse",
    "cTypeGPeGPe": "diffuse",
    "cTypeGPeGPi": "diffuse",
    "cTypeGPeMSN": "diffuse",
    "cTypeGPeSTN": "focused",
    "cTypeMSNArky": "focused",
    "cTypeMSNGPe": "focused",
    "cTypeMSNGPi": "focused",
    "cTypeMSNMSN": "diffuse",
    "cTypeMSNProt": "focused",
    "cTypePTNFSI": "focused",
    "cTypePTNMSN": "focused",
    "cTypePTNSTN": "focused",
    "cTypeProtArky": "diffuse",
    "cTypeProtGPi": "diffuse",
    "cTypeProtProt": "diffuse",
    "cTypeProtSTN": "focused",
    "cTypeSTNArky": "diffuse",
    "cTypeSTNFSI": "diffuse",
    "cTypeSTNGPe": "diffuse",
    "cTypeSTNGPi": "diffuse",
    "cTypeSTNMSN": "diffuse",
    "cTypeSTNProt": "diffuse",
    "durationH": "08",
    "email": "",
    "nbArky": 5.0,
    "nbCMPf": 9.0,
    "nbCSN": 3000.0,
    "nbCh": 1,
    "nbFSI": 53.0,
    "nbGPe": 25.0,
    "nbGPi": 14.0,
    "nbMSN": 2644.0,
    "nbPTN": 100.0,
    "nbProt": 20.0,
    "nbSTN": 8.0,
    "nbcpu": 8,
    "nbnodes": "1",
    "nestSeed": 20,
    "parrotCMPf": True,
    "platform": "Local",
    "pythonSeed": 10,
    "redundancyArkyArky": 3,
    "redundancyArkyFSI": 3,
    "redundancyArkyMSN": 3,
    "redundancyArkyProt": 3,
    "redundancyCMPfArky": 3,
    "redundancyCMPfFSI": 3,
    "redundancyCMPfGPe": 3,
    "redundancyCMPfGPi": 3,
    "redundancyCMPfMSN": 3,
    "redundancyCMPfProt": 3,
    "redundancyCMPfSTN": 3,
    "redundancyCSNFSI": 3,
    "redundancyCSNMSN": 3,
    "redundancyFSIFSI": 3,
    "redundancyFSIMSN": 3,
    "redundancyGPeFSI": 3,
    "redundancyGPeGPe": 3,
    "redundancyGPeGPi": 3,
    "redundancyGPeMSN": 3,
    "redundancyGPeSTN": 3,
    "redundancyMSNArky": 3,
    "redundancyMSNGPe": 3,
    "redundancyMSNGPi": 3,
    "redundancyMSNMSN": 3,
    "redundancyMSNProt": 3,
    "redundancyPTNFSI": 3,
    "redundancyPTNMSN": 3,
    "redundancyPTNSTN": 3,
    "redundancyProtArky": 3,
    "redundancyProtGPi": 3,
    "redundancyProtProt": 3,
    "redundancyProtSTN": 3,
    "redundancySTNArky": 3,
    "redundancySTNFSI": 3,
    "redundancySTNGPe": 3,
    "redundancySTNGPi": 3,
    "redundancySTNMSN": 3,
    "redundancySTNProt": 3,
    "simulator": "Nengo",
    "splitGPe": False,
    "stochastic_delays": None,
    "tSimu": 5000.0,
    "whichTest": "sBCBG"
}

interactive = False

storeGDF = False