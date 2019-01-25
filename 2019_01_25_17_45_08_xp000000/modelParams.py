#!/apps/free/python/2.7.10/bin/python 

## This file was auto-generated by run.py called with the following arguments:
# run.py --platform Local --whichTest sBCBG --nbcpu -1 --nbCh 2 --custom baseParams.py --simulator Nengo

## ID string of experiment:
# 2019_01_25_17_45_08_xp000000

## Reproducibility info:
#  platform = Local
#  git commit ID = c8987ff3c06fd41363cda6d1fd9bb4cca402bd9c
#  Changes not yet commited in the following files:  D 2019_01_07_17_35_43_xp000000/PoissonGenerator.py -  D 2019_01_07_17_35_43_xp000000/PoissonGenerator.pyc -  D 2019_01_07_17_35_43_xp000000/__init__.py -  D 2019_01_07_17_35_43_xp000000/filter.py -  D 2019_01_07_17_35_43_xp000000/modelParams.py -  D 2019_01_07_17_35_43_xp000000/modelParams.pyc -  D 2019_01_07_17_35_43_xp000000/nstrand.py -  D 2019_01_07_17_35_43_xp000000/sBCBG.py -  D 2019_01_07_17_35_43_xp000000/solutions_simple_unique.csv -  D 2019_01_07_17_35_43_xp000000/spikeProcessing.py -  D 2019_01_07_17_36_03_xp000000/PoissonGenerator.py -  D 2019_01_07_17_36_03_xp000000/PoissonGenerator.pyc -  D 2019_01_07_17_36_03_xp000000/__init__.py -  D 2019_01_07_17_36_03_xp000000/filter.py -  D 2019_01_07_17_36_03_xp000000/log/OutSummary.txt -  D 2019_01_07_17_36_03_xp000000/log/firingRates.csv -  D 2019_01_07_17_36_03_xp000000/modelParams.py -  D 2019_01_07_17_36_03_xp000000/modelParams.pyc -  D 2019_01_07_17_36_03_xp000000/nstrand.py -  D 2019_01_07_17_36_03_xp000000/params_score.csv -  D 2019_01_07_17_36_03_xp000000/plots/CMPf outputs.png -  D 2019_01_07_17_36_03_xp000000/plots/CSN outputs.png -  D 2019_01_07_17_36_03_xp000000/plots/FSI inputs.png -  D 2019_01_07_17_36_03_xp000000/plots/FSI multimeters.png -  D 2019_01_07_17_36_03_xp000000/plots/FSI outputs.png -  D 2019_01_07_17_36_03_xp000000/plots/GPe inputs.png -  D 2019_01_07_17_36_03_xp000000/plots/GPe multimeters.png -  D 2019_01_07_17_36_03_xp000000/plots/GPe outputs.png -  D 2019_01_07_17_36_03_xp000000/plots/GPi inputs.png -  D 2019_01_07_17_36_03_xp000000/plots/GPi multimeters.png -  D 2019_01_07_17_36_03_xp000000/plots/GPi outputs.png -  D 2019_01_07_17_36_03_xp000000/plots/MSN inputs.png -  D 2019_01_07_17_36_03_xp000000/plots/MSN multimeters.png -  D 2019_01_07_17_36_03_xp000000/plots/MSN outputs.png -  D 2019_01_07_17_36_03_xp000000/plots/PTN outputs.png -  D 2019_01_07_17_36_03_xp000000/plots/STN inputs.png -  D 2019_01_07_17_36_03_xp000000/plots/STN multimeters.png -  D 2019_01_07_17_36_03_xp000000/plots/STN outputs.png -  D 2019_01_07_17_36_03_xp000000/sBCBG.py -  D 2019_01_07_17_36_03_xp000000/score.txt -  D 2019_01_07_17_36_03_xp000000/solutions_simple_unique.csv -  D 2019_01_07_17_36_03_xp000000/spikeProcessing.py -  D 2019_01_07_17_36_03_xp000000/validationArray.csv -  D 2019_01_07_17_37_26_xp000000/PoissonGenerator.py -  D 2019_01_07_17_37_26_xp000000/PoissonGenerator.pyc -  D 2019_01_07_17_37_26_xp000000/__init__.py -  D 2019_01_07_17_37_26_xp000000/filter.py -  D 2019_01_07_17_37_26_xp000000/log/OutSummary.txt -  D 2019_01_07_17_37_26_xp000000/log/firingRates.csv -  D 2019_01_07_17_37_26_xp000000/modelParams.py -  D 2019_01_07_17_37_26_xp000000/modelParams.pyc -  D 2019_01_07_17_37_26_xp000000/nstrand.py -  D 2019_01_07_17_37_26_xp000000/params_score.csv -  D 2019_01_07_17_37_26_xp000000/plots/CMPf outputs.png -  D 2019_01_07_17_37_26_xp000000/plots/CSN outputs.png -  D 2019_01_07_17_37_26_xp000000/plots/FSI inputs.png -  D 2019_01_07_17_37_26_xp000000/plots/FSI multimeters.png -  D 2019_01_07_17_37_26_xp000000/plots/FSI outputs.png -  D 2019_01_07_17_37_26_xp000000/plots/GPe inputs.png -  D 2019_01_07_17_37_26_xp000000/plots/GPe multimeters.png -  D 2019_01_07_17_37_26_xp000000/plots/GPe outputs.png -  D 2019_01_07_17_37_26_xp000000/plots/GPi inputs.png -  D 2019_01_07_17_37_26_xp000000/plots/GPi multimeters.png -  D 2019_01_07_17_37_26_xp000000/plots/GPi outputs.png -  D 2019_01_07_17_37_26_xp000000/plots/MSN inputs.png -  D 2019_01_07_17_37_26_xp000000/plots/MSN multimeters.png -  D 2019_01_07_17_37_26_xp000000/plots/MSN outputs.png -  D 2019_01_07_17_37_26_xp000000/plots/PTN outputs.png -  D 2019_01_07_17_37_26_xp000000/plots/STN inputs.png -  D 2019_01_07_17_37_26_xp000000/plots/STN multimeters.png -  D 2019_01_07_17_37_26_xp000000/plots/STN outputs.png -  D 2019_01_07_17_37_26_xp000000/sBCBG.py -  D 2019_01_07_17_37_26_xp000000/score.txt -  D 2019_01_07_17_37_26_xp000000/solutions_simple_unique.csv -  D 2019_01_07_17_37_26_xp000000/spikeProcessing.py -  D 2019_01_07_17_37_26_xp000000/validationArray.csv -  D 2019_01_07_17_40_24_xp000000/PoissonGenerator.py -  D 2019_01_07_17_40_24_xp000000/__init__.py -  D 2019_01_07_17_40_24_xp000000/filter.py -  D 2019_01_07_17_40_24_xp000000/modelParams.py -  D 2019_01_07_17_40_24_xp000000/nstrand.py -  D 2019_01_07_17_40_24_xp000000/sBCBG.py -  D 2019_01_07_17_40_24_xp000000/solutions_simple_unique.csv -  D 2019_01_07_17_40_24_xp000000/spikeProcessing.py -  D 2019_01_07_17_40_49_xp000000/PoissonGenerator.py -  D 2019_01_07_17_40_49_xp000000/PoissonGenerator.pyc -  D 2019_01_07_17_40_49_xp000000/__init__.py -  D 2019_01_07_17_40_49_xp000000/filter.py -  D 2019_01_07_17_40_49_xp000000/modelParams.py -  D 2019_01_07_17_40_49_xp000000/modelParams.pyc -  D 2019_01_07_17_40_49_xp000000/nstrand.py -  D 2019_01_07_17_40_49_xp000000/sBCBG.py -  D 2019_01_07_17_40_49_xp000000/solutions_simple_unique.csv -  D 2019_01_07_17_40_49_xp000000/spikeProcessing.py -  D 2019_01_07_17_42_56_xp000000/PoissonGenerator.py -  D 2019_01_07_17_42_56_xp000000/PoissonGenerator.pyc -  D 2019_01_07_17_42_56_xp000000/__init__.py -  D 2019_01_07_17_42_56_xp000000/filter.py -  D 2019_01_07_17_42_56_xp000000/modelParams.py -  D 2019_01_07_17_42_56_xp000000/modelParams.pyc -  D 2019_01_07_17_42_56_xp000000/nstrand.py -  D 2019_01_07_17_42_56_xp000000/sBCBG.py -  D 2019_01_07_17_42_56_xp000000/solutions_simple_unique.csv -  D 2019_01_07_17_42_56_xp000000/spikeProcessing.py -  D 2019_01_07_17_43_36_xp000000/PoissonGenerator.py -  D 2019_01_07_17_43_36_xp000000/PoissonGenerator.pyc -  D 2019_01_07_17_43_36_xp000000/__init__.py -  D 2019_01_07_17_43_36_xp000000/filter.py -  D 2019_01_07_17_43_36_xp000000/log/OutSummary.txt -  D 2019_01_07_17_43_36_xp000000/log/firingRates.csv -  D 2019_01_07_17_43_36_xp000000/modelParams.py -  D 2019_01_07_17_43_36_xp000000/modelParams.pyc -  D 2019_01_07_17_43_36_xp000000/nstrand.py -  D 2019_01_07_17_43_36_xp000000/params_score.csv -  D 2019_01_07_17_43_36_xp000000/plots/CMPf outputs.png -  D 2019_01_07_17_43_36_xp000000/plots/CSN outputs.png -  D 2019_01_07_17_43_36_xp000000/plots/FSI inputs.png -  D 2019_01_07_17_43_36_xp000000/plots/FSI multimeters.png -  D 2019_01_07_17_43_36_xp000000/plots/FSI outputs.png -  D 2019_01_07_17_43_36_xp000000/plots/GPe inputs.png -  D 2019_01_07_17_43_36_xp000000/plots/GPe multimeters.png -  D 2019_01_07_17_43_36_xp000000/plots/GPe outputs.png -  D 2019_01_07_17_43_36_xp000000/plots/GPi inputs.png -  D 2019_01_07_17_43_36_xp000000/plots/GPi multimeters.png -  D 2019_01_07_17_43_36_xp000000/plots/GPi outputs.png -  D 2019_01_07_17_43_36_xp000000/plots/MSN inputs.png -  D 2019_01_07_17_43_36_xp000000/plots/MSN multimeters.png -  D 2019_01_07_17_43_36_xp000000/plots/MSN outputs.png -  D 2019_01_07_17_43_36_xp000000/plots/PTN outputs.png -  D 2019_01_07_17_43_36_xp000000/plots/STN inputs.png -  D 2019_01_07_17_43_36_xp000000/plots/STN multimeters.png -  D 2019_01_07_17_43_36_xp000000/plots/STN outputs.png -  D 2019_01_07_17_43_36_xp000000/sBCBG.py -  D 2019_01_07_17_43_36_xp000000/score.txt -  D 2019_01_07_17_43_36_xp000000/solutions_simple_unique.csv -  D 2019_01_07_17_43_36_xp000000/spikeProcessing.py -  D 2019_01_07_17_43_36_xp000000/validationArray.csv -  D 2019_01_07_17_46_27_xp000000/PoissonGenerator.py -  D 2019_01_07_17_46_27_xp000000/PoissonGenerator.pyc -  D 2019_01_07_17_46_27_xp000000/__init__.py -  D 2019_01_07_17_46_27_xp000000/filter.py -  D 2019_01_07_17_46_27_xp000000/log/OutSummary.txt -  D 2019_01_07_17_46_27_xp000000/log/firingRates.csv -  D 2019_01_07_17_46_27_xp000000/modelParams.py -  D 2019_01_07_17_46_27_xp000000/modelParams.pyc -  D 2019_01_07_17_46_27_xp000000/nstrand.py -  D 2019_01_07_17_46_27_xp000000/params_score.csv -  D 2019_01_07_17_46_27_xp000000/plots/CMPf outputs.png -  D 2019_01_07_17_46_27_xp000000/plots/CSN outputs.png -  D 2019_01_07_17_46_27_xp000000/plots/FSI inputs.png -  D 2019_01_07_17_46_27_xp000000/plots/FSI multimeters.png -  D 2019_01_07_17_46_27_xp000000/plots/FSI outputs.png -  D 2019_01_07_17_46_27_xp000000/plots/GPe inputs.png -  D 2019_01_07_17_46_27_xp000000/plots/GPe multimeters.png -  D 2019_01_07_17_46_27_xp000000/plots/GPe outputs.png -  D 2019_01_07_17_46_27_xp000000/plots/GPi inputs.png -  D 2019_01_07_17_46_27_xp000000/plots/GPi multimeters.png -  D 2019_01_07_17_46_27_xp000000/plots/GPi outputs.png -  D 2019_01_07_17_46_27_xp000000/plots/MSN inputs.png -  D 2019_01_07_17_46_27_xp000000/plots/MSN multimeters.png -  D 2019_01_07_17_46_27_xp000000/plots/MSN outputs.png -  D 2019_01_07_17_46_27_xp000000/plots/PTN outputs.png -  D 2019_01_07_17_46_27_xp000000/plots/STN inputs.png -  D 2019_01_07_17_46_27_xp000000/plots/STN multimeters.png -  D 2019_01_07_17_46_27_xp000000/plots/STN outputs.png -  D 2019_01_07_17_46_27_xp000000/sBCBG.py -  D 2019_01_07_17_46_27_xp000000/score.txt -  D 2019_01_07_17_46_27_xp000000/solutions_simple_unique.csv -  D 2019_01_07_17_46_27_xp000000/spikeProcessing.py -  D 2019_01_07_17_46_27_xp000000/validationArray.csv -  D 2019_01_11_21_42_45_xp000000/PoissonGenerator.py -  D 2019_01_11_21_42_45_xp000000/PoissonGenerator.pyc -  D 2019_01_11_21_42_45_xp000000/__init__.py -  D 2019_01_11_21_42_45_xp000000/filter.py -  D 2019_01_11_21_42_45_xp000000/modelParams.py -  D 2019_01_11_21_42_45_xp000000/modelParams.pyc -  D 2019_01_11_21_42_45_xp000000/nstrand.py -  D 2019_01_11_21_42_45_xp000000/sBCBG.py -  D 2019_01_11_21_42_45_xp000000/solutions_simple_unique.csv -  D 2019_01_11_21_42_45_xp000000/spikeProcessing.py -  D 2019_01_11_21_43_25_xp000000/PoissonGenerator.py -  D 2019_01_11_21_43_25_xp000000/PoissonGenerator.pyc -  D 2019_01_11_21_43_25_xp000000/__init__.py -  D 2019_01_11_21_43_25_xp000000/filter.py -  D 2019_01_11_21_43_25_xp000000/modelParams.py -  D 2019_01_11_21_43_25_xp000000/modelParams.pyc -  D 2019_01_11_21_43_25_xp000000/nstrand.py -  D 2019_01_11_21_43_25_xp000000/sBCBG.py -  D 2019_01_11_21_43_25_xp000000/solutions_simple_unique.csv -  D 2019_01_11_21_43_25_xp000000/spikeProcessing.py -  D 2019_01_11_21_43_55_xp000000/PoissonGenerator.py -  D 2019_01_11_21_43_55_xp000000/PoissonGenerator.pyc -  D 2019_01_11_21_43_55_xp000000/__init__.py -  D 2019_01_11_21_43_55_xp000000/filter.py -  D 2019_01_11_21_43_55_xp000000/modelParams.py -  D 2019_01_11_21_43_55_xp000000/modelParams.pyc -  D 2019_01_11_21_43_55_xp000000/nstrand.py -  D 2019_01_11_21_43_55_xp000000/sBCBG.py -  D 2019_01_11_21_43_55_xp000000/solutions_simple_unique.csv -  D 2019_01_11_21_43_55_xp000000/spikeProcessing.py -  D 2019_01_11_21_47_39_xp000000/PoissonGenerator.py -  D 2019_01_11_21_47_39_xp000000/PoissonGenerator.pyc -  D 2019_01_11_21_47_39_xp000000/__init__.py -  D 2019_01_11_21_47_39_xp000000/filter.py -  D 2019_01_11_21_47_39_xp000000/modelParams.py -  D 2019_01_11_21_47_39_xp000000/modelParams.pyc -  D 2019_01_11_21_47_39_xp000000/nstrand.py -  D 2019_01_11_21_47_39_xp000000/sBCBG.py -  D 2019_01_11_21_47_39_xp000000/solutions_simple_unique.csv -  D 2019_01_11_21_47_39_xp000000/spikeProcessing.py -  D 2019_01_11_21_47_55_xp000000/PoissonGenerator.py -  D 2019_01_11_21_47_55_xp000000/PoissonGenerator.pyc -  D 2019_01_11_21_47_55_xp000000/__init__.py -  D 2019_01_11_21_47_55_xp000000/filter.py -  D 2019_01_11_21_47_55_xp000000/modelParams.py -  D 2019_01_11_21_47_55_xp000000/modelParams.pyc -  D 2019_01_11_21_47_55_xp000000/nstrand.py -  D 2019_01_11_21_47_55_xp000000/sBCBG.py -  D 2019_01_11_21_47_55_xp000000/solutions_simple_unique.csv -  D 2019_01_11_21_47_55_xp000000/spikeProcessing.py -  D __pycache__/PoissonGenerator.cpython-35.pyc -  D __pycache__/modelParams.cpython-35.pyc -  M baseParams.py -  M baseParams.pyc -  M nengo_bg.py -  M sBCBG.py - 

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
    "IeArky": 0.0,
    "IeFSI": 1.0,
    "IeGPe": 15.0,
    "IeGPi": 15.0,
    "IeMSN": 29.75,
    "IeProt": 0.0,
    "IeSTN": 9.25,
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
    "nbCMPf": 10.0,
    "nbCSN": 10.0,
    "nbCh": 2,
    "nbFSI": 53.0,
    "nbGPe": 25.0,
    "nbGPi": 14.0,
    "nbMSN": 2644.0,
    "nbPTN": 10.0,
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