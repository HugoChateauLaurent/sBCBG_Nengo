#!/apps/free/python/2.7.10/bin/python 

## This file was auto-generated by run.py called with the following arguments:
# run.py --platform Local --whichTest sBCBG --nbcpu -1 --custom baseParams.py --simulator Nengo

## ID string of experiment:
# 2018_12_06_20_49_01_xp000000

## Reproducibility info:
#  platform = Local
#  git commit ID = adda6b3da666b4448b53c292fb1490f6e75fad55
#  Changes not yet commited in the following files:  D 2018_12_01_16_54_07_xp000000/LGneurons.py -  D 2018_12_01_16_54_07_xp000000/LGneurons.pyc -  D 2018_12_01_16_54_07_xp000000/__init__.py -  D 2018_12_01_16_54_07_xp000000/filter.py -  D 2018_12_01_16_54_07_xp000000/filter.pyc -  D 2018_12_01_16_54_07_xp000000/iniBG.py -  D 2018_12_01_16_54_07_xp000000/iniBG.pyc -  D 2018_12_01_16_54_07_xp000000/modelParams.py -  D 2018_12_01_16_54_07_xp000000/modelParams.pyc -  D 2018_12_01_16_54_07_xp000000/nstrand.py -  D 2018_12_01_16_54_07_xp000000/nstrand.pyc -  D 2018_12_01_16_54_07_xp000000/solutions_simple_unique.csv -  D 2018_12_01_16_54_07_xp000000/spikeProcessing.py -  D 2018_12_01_16_54_07_xp000000/spikeProcessing.pyc -  D 2018_12_01_16_54_07_xp000000/testPlausibility.py -  D 2018_12_01_16_54_41_xp000000/LGneurons.py -  D 2018_12_01_16_54_41_xp000000/LGneurons.pyc -  D 2018_12_01_16_54_41_xp000000/__init__.py -  D 2018_12_01_16_54_41_xp000000/filter.py -  D 2018_12_01_16_54_41_xp000000/filter.pyc -  D 2018_12_01_16_54_41_xp000000/iniBG.py -  D 2018_12_01_16_54_41_xp000000/iniBG.pyc -  D 2018_12_01_16_54_41_xp000000/modelParams.py -  D 2018_12_01_16_54_41_xp000000/modelParams.pyc -  D 2018_12_01_16_54_41_xp000000/nstrand.py -  D 2018_12_01_16_54_41_xp000000/nstrand.pyc -  D 2018_12_01_16_54_41_xp000000/solutions_simple_unique.csv -  D 2018_12_01_16_54_41_xp000000/spikeProcessing.py -  D 2018_12_01_16_54_41_xp000000/spikeProcessing.pyc -  D 2018_12_01_16_54_41_xp000000/testPlausibility.py -  D 2018_12_01_16_54_47_xp000000/LGneurons.py -  D 2018_12_01_16_54_47_xp000000/LGneurons.pyc -  D 2018_12_01_16_54_47_xp000000/__init__.py -  D 2018_12_01_16_54_47_xp000000/filter.py -  D 2018_12_01_16_54_47_xp000000/filter.pyc -  D 2018_12_01_16_54_47_xp000000/iniBG.py -  D 2018_12_01_16_54_47_xp000000/iniBG.pyc -  D 2018_12_01_16_54_47_xp000000/modelParams.py -  D 2018_12_01_16_54_47_xp000000/modelParams.pyc -  D 2018_12_01_16_54_47_xp000000/nstrand.py -  D 2018_12_01_16_54_47_xp000000/nstrand.pyc -  D 2018_12_01_16_54_47_xp000000/solutions_simple_unique.csv -  D 2018_12_01_16_54_47_xp000000/spikeProcessing.py -  D 2018_12_01_16_54_47_xp000000/spikeProcessing.pyc -  D 2018_12_01_16_54_47_xp000000/testPlausibility.py -  D 2018_12_01_16_54_57_xp000000/LGneurons.py -  D 2018_12_01_16_54_57_xp000000/LGneurons.pyc -  D 2018_12_01_16_54_57_xp000000/__init__.py -  D 2018_12_01_16_54_57_xp000000/filter.py -  D 2018_12_01_16_54_57_xp000000/filter.pyc -  D 2018_12_01_16_54_57_xp000000/iniBG.py -  D 2018_12_01_16_54_57_xp000000/iniBG.pyc -  D 2018_12_01_16_54_57_xp000000/modelParams.py -  D 2018_12_01_16_54_57_xp000000/modelParams.pyc -  D 2018_12_01_16_54_57_xp000000/nstrand.py -  D 2018_12_01_16_54_57_xp000000/nstrand.pyc -  D 2018_12_01_16_54_57_xp000000/solutions_simple_unique.csv -  D 2018_12_01_16_54_57_xp000000/spikeProcessing.py -  D 2018_12_01_16_54_57_xp000000/spikeProcessing.pyc -  D 2018_12_01_16_54_57_xp000000/testPlausibility.py -  D 2018_12_01_16_55_13_xp000000/LGneurons.py -  D 2018_12_01_16_55_13_xp000000/LGneurons.pyc -  D 2018_12_01_16_55_13_xp000000/__init__.py -  D 2018_12_01_16_55_13_xp000000/filter.py -  D 2018_12_01_16_55_13_xp000000/filter.pyc -  D 2018_12_01_16_55_13_xp000000/iniBG.py -  D 2018_12_01_16_55_13_xp000000/iniBG.pyc -  D 2018_12_01_16_55_13_xp000000/modelParams.py -  D 2018_12_01_16_55_13_xp000000/modelParams.pyc -  D 2018_12_01_16_55_13_xp000000/nstrand.py -  D 2018_12_01_16_55_13_xp000000/nstrand.pyc -  D 2018_12_01_16_55_13_xp000000/solutions_simple_unique.csv -  D 2018_12_01_16_55_13_xp000000/spikeProcessing.py -  D 2018_12_01_16_55_13_xp000000/spikeProcessing.pyc -  D 2018_12_01_16_55_13_xp000000/testPlausibility.py -  D 2018_12_01_16_55_22_xp000000/LGneurons.py -  D 2018_12_01_16_55_22_xp000000/LGneurons.pyc -  D 2018_12_01_16_55_22_xp000000/__init__.py -  D 2018_12_01_16_55_22_xp000000/filter.py -  D 2018_12_01_16_55_22_xp000000/filter.pyc -  D 2018_12_01_16_55_22_xp000000/iniBG.py -  D 2018_12_01_16_55_22_xp000000/iniBG.pyc -  D 2018_12_01_16_55_22_xp000000/modelParams.py -  D 2018_12_01_16_55_22_xp000000/modelParams.pyc -  D 2018_12_01_16_55_22_xp000000/nstrand.py -  D 2018_12_01_16_55_22_xp000000/nstrand.pyc -  D 2018_12_01_16_55_22_xp000000/solutions_simple_unique.csv -  D 2018_12_01_16_55_22_xp000000/spikeProcessing.py -  D 2018_12_01_16_55_22_xp000000/spikeProcessing.pyc -  D 2018_12_01_16_55_22_xp000000/testPlausibility.py -  D 2018_12_01_16_55_29_xp000000/LGneurons.py -  D 2018_12_01_16_55_29_xp000000/LGneurons.pyc -  D 2018_12_01_16_55_29_xp000000/__init__.py -  D 2018_12_01_16_55_29_xp000000/filter.py -  D 2018_12_01_16_55_29_xp000000/filter.pyc -  D 2018_12_01_16_55_29_xp000000/iniBG.py -  D 2018_12_01_16_55_29_xp000000/iniBG.pyc -  D 2018_12_01_16_55_29_xp000000/modelParams.py -  D 2018_12_01_16_55_29_xp000000/modelParams.pyc -  D 2018_12_01_16_55_29_xp000000/nstrand.py -  D 2018_12_01_16_55_29_xp000000/nstrand.pyc -  D 2018_12_01_16_55_29_xp000000/solutions_simple_unique.csv -  D 2018_12_01_16_55_29_xp000000/spikeProcessing.py -  D 2018_12_01_16_55_29_xp000000/spikeProcessing.pyc -  D 2018_12_01_16_55_29_xp000000/testPlausibility.py -  D 2018_12_01_16_56_03_xp000000/LGneurons.py -  D 2018_12_01_16_56_03_xp000000/LGneurons.pyc -  D 2018_12_01_16_56_03_xp000000/__init__.py -  D 2018_12_01_16_56_03_xp000000/filter.py -  D 2018_12_01_16_56_03_xp000000/filter.pyc -  D 2018_12_01_16_56_03_xp000000/iniBG.py -  D 2018_12_01_16_56_03_xp000000/iniBG.pyc -  D 2018_12_01_16_56_03_xp000000/modelParams.py -  D 2018_12_01_16_56_03_xp000000/modelParams.pyc -  D 2018_12_01_16_56_03_xp000000/nstrand.py -  D 2018_12_01_16_56_03_xp000000/nstrand.pyc -  D 2018_12_01_16_56_03_xp000000/solutions_simple_unique.csv -  D 2018_12_01_16_56_03_xp000000/spikeProcessing.py -  D 2018_12_01_16_56_03_xp000000/spikeProcessing.pyc -  D 2018_12_01_16_56_03_xp000000/testPlausibility.py -  D 2018_12_01_17_03_05_xp000000/LGneurons.py -  D 2018_12_01_17_03_05_xp000000/__init__.py -  D 2018_12_01_17_03_05_xp000000/filter.py -  D 2018_12_01_17_03_05_xp000000/iniBG.py -  D 2018_12_01_17_03_05_xp000000/iniBG.pyc -  D 2018_12_01_17_03_05_xp000000/modelParams.py -  D 2018_12_01_17_03_05_xp000000/nstrand.py -  D 2018_12_01_17_03_05_xp000000/nstrand.pyc -  D 2018_12_01_17_03_05_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_03_05_xp000000/spikeProcessing.py -  D 2018_12_01_17_03_05_xp000000/testPlausibility.py -  D 2018_12_01_17_05_16_xp000000/LGneurons.py -  D 2018_12_01_17_05_16_xp000000/__init__.py -  D 2018_12_01_17_05_16_xp000000/filter.py -  D 2018_12_01_17_05_16_xp000000/iniBG.py -  D 2018_12_01_17_05_16_xp000000/iniBG.pyc -  D 2018_12_01_17_05_16_xp000000/modelParams.py -  D 2018_12_01_17_05_16_xp000000/nstrand.py -  D 2018_12_01_17_05_16_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_05_16_xp000000/spikeProcessing.py -  D 2018_12_01_17_05_16_xp000000/testPlausibility.py -  D 2018_12_01_17_05_37_xp000000/LGneurons.py -  D 2018_12_01_17_05_37_xp000000/__init__.py -  D 2018_12_01_17_05_37_xp000000/filter.py -  D 2018_12_01_17_05_37_xp000000/iniBG.py -  D 2018_12_01_17_05_37_xp000000/iniBG.pyc -  D 2018_12_01_17_05_37_xp000000/modelParams.py -  D 2018_12_01_17_05_37_xp000000/nstrand.py -  D 2018_12_01_17_05_37_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_05_37_xp000000/spikeProcessing.py -  D 2018_12_01_17_05_37_xp000000/testPlausibility.py -  D 2018_12_01_17_05_44_xp000000/LGneurons.py -  D 2018_12_01_17_05_44_xp000000/__init__.py -  D 2018_12_01_17_05_44_xp000000/filter.py -  D 2018_12_01_17_05_44_xp000000/iniBG.py -  D 2018_12_01_17_05_44_xp000000/iniBG.pyc -  D 2018_12_01_17_05_44_xp000000/modelParams.py -  D 2018_12_01_17_05_44_xp000000/nstrand.py -  D 2018_12_01_17_05_44_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_05_44_xp000000/spikeProcessing.py -  D 2018_12_01_17_05_44_xp000000/testPlausibility.py -  D 2018_12_01_17_05_57_xp000000/LGneurons.py -  D 2018_12_01_17_05_57_xp000000/__init__.py -  D 2018_12_01_17_05_57_xp000000/filter.py -  D 2018_12_01_17_05_57_xp000000/iniBG.py -  D 2018_12_01_17_05_57_xp000000/iniBG.pyc -  D 2018_12_01_17_05_57_xp000000/modelParams.py -  D 2018_12_01_17_05_57_xp000000/nstrand.py -  D 2018_12_01_17_05_57_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_05_57_xp000000/spikeProcessing.py -  D 2018_12_01_17_05_57_xp000000/testPlausibility.py -  D 2018_12_01_17_06_06_xp000000/LGneurons.py -  D 2018_12_01_17_06_06_xp000000/__init__.py -  D 2018_12_01_17_06_06_xp000000/filter.py -  D 2018_12_01_17_06_06_xp000000/iniBG.py -  D 2018_12_01_17_06_06_xp000000/iniBG.pyc -  D 2018_12_01_17_06_06_xp000000/modelParams.py -  D 2018_12_01_17_06_06_xp000000/nstrand.py -  D 2018_12_01_17_06_06_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_06_06_xp000000/spikeProcessing.py -  D 2018_12_01_17_06_06_xp000000/testPlausibility.py -  D 2018_12_01_17_06_32_xp000000/LGneurons.py -  D 2018_12_01_17_06_32_xp000000/__init__.py -  D 2018_12_01_17_06_32_xp000000/filter.py -  D 2018_12_01_17_06_32_xp000000/iniBG.py -  D 2018_12_01_17_06_32_xp000000/iniBG.pyc -  D 2018_12_01_17_06_32_xp000000/modelParams.py -  D 2018_12_01_17_06_32_xp000000/nstrand.py -  D 2018_12_01_17_06_32_xp000000/nstrand.pyc -  D 2018_12_01_17_06_32_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_06_32_xp000000/spikeProcessing.py -  D 2018_12_01_17_06_32_xp000000/testPlausibility.py -  D 2018_12_01_17_06_39_xp000000/LGneurons.py -  D 2018_12_01_17_06_39_xp000000/LGneurons.pyc -  D 2018_12_01_17_06_39_xp000000/__init__.py -  D 2018_12_01_17_06_39_xp000000/filter.py -  D 2018_12_01_17_06_39_xp000000/filter.pyc -  D 2018_12_01_17_06_39_xp000000/iniBG.py -  D 2018_12_01_17_06_39_xp000000/iniBG.pyc -  D 2018_12_01_17_06_39_xp000000/modelParams.py -  D 2018_12_01_17_06_39_xp000000/modelParams.pyc -  D 2018_12_01_17_06_39_xp000000/nstrand.py -  D 2018_12_01_17_06_39_xp000000/nstrand.pyc -  D 2018_12_01_17_06_39_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_06_39_xp000000/spikeProcessing.py -  D 2018_12_01_17_06_39_xp000000/spikeProcessing.pyc -  D 2018_12_01_17_06_39_xp000000/testPlausibility.py -  D 2018_12_01_17_06_52_xp000000/LGneurons.py -  D 2018_12_01_17_06_52_xp000000/LGneurons.pyc -  D 2018_12_01_17_06_52_xp000000/__init__.py -  D 2018_12_01_17_06_52_xp000000/filter.py -  D 2018_12_01_17_06_52_xp000000/filter.pyc -  D 2018_12_01_17_06_52_xp000000/iniBG.py -  D 2018_12_01_17_06_52_xp000000/iniBG.pyc -  D 2018_12_01_17_06_52_xp000000/modelParams.py -  D 2018_12_01_17_06_52_xp000000/modelParams.pyc -  D 2018_12_01_17_06_52_xp000000/nstrand.py -  D 2018_12_01_17_06_52_xp000000/nstrand.pyc -  D 2018_12_01_17_06_52_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_06_52_xp000000/spikeProcessing.py -  D 2018_12_01_17_06_52_xp000000/spikeProcessing.pyc -  D 2018_12_01_17_06_52_xp000000/testPlausibility.py -  D 2018_12_01_17_07_56_xp000000/LGneurons.py -  D 2018_12_01_17_07_56_xp000000/LGneurons.pyc -  D 2018_12_01_17_07_56_xp000000/__init__.py -  D 2018_12_01_17_07_56_xp000000/filter.py -  D 2018_12_01_17_07_56_xp000000/filter.pyc -  D 2018_12_01_17_07_56_xp000000/iniBG.py -  D 2018_12_01_17_07_56_xp000000/iniBG.pyc -  D 2018_12_01_17_07_56_xp000000/modelParams.py -  D 2018_12_01_17_07_56_xp000000/modelParams.pyc -  D 2018_12_01_17_07_56_xp000000/nstrand.py -  D 2018_12_01_17_07_56_xp000000/nstrand.pyc -  D 2018_12_01_17_07_56_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_07_56_xp000000/spikeProcessing.py -  D 2018_12_01_17_07_56_xp000000/spikeProcessing.pyc -  D 2018_12_01_17_07_56_xp000000/testPlausibility.py -  D 2018_12_01_17_10_05_xp000000/LGneurons.py -  D 2018_12_01_17_10_05_xp000000/LGneurons.pyc -  D 2018_12_01_17_10_05_xp000000/__init__.py -  D 2018_12_01_17_10_05_xp000000/filter.py -  D 2018_12_01_17_10_05_xp000000/filter.pyc -  D 2018_12_01_17_10_05_xp000000/iniBG.py -  D 2018_12_01_17_10_05_xp000000/iniBG.pyc -  D 2018_12_01_17_10_05_xp000000/modelParams.py -  D 2018_12_01_17_10_05_xp000000/modelParams.pyc -  D 2018_12_01_17_10_05_xp000000/nstrand.py -  D 2018_12_01_17_10_05_xp000000/nstrand.pyc -  D 2018_12_01_17_10_05_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_10_05_xp000000/spikeProcessing.py -  D 2018_12_01_17_10_05_xp000000/spikeProcessing.pyc -  D 2018_12_01_17_10_05_xp000000/testPlausibility.py -  D 2018_12_01_17_38_52_xp000000/LGneurons.py -  D 2018_12_01_17_38_52_xp000000/LGneurons.pyc -  D 2018_12_01_17_38_52_xp000000/__init__.py -  D 2018_12_01_17_38_52_xp000000/filter.py -  D 2018_12_01_17_38_52_xp000000/filter.pyc -  D 2018_12_01_17_38_52_xp000000/iniBG.py -  D 2018_12_01_17_38_52_xp000000/iniBG.pyc -  D 2018_12_01_17_38_52_xp000000/modelParams.py -  D 2018_12_01_17_38_52_xp000000/modelParams.pyc -  D 2018_12_01_17_38_52_xp000000/nstrand.py -  D 2018_12_01_17_38_52_xp000000/nstrand.pyc -  D 2018_12_01_17_38_52_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_38_52_xp000000/spikeProcessing.py -  D 2018_12_01_17_38_52_xp000000/spikeProcessing.pyc -  D 2018_12_01_17_38_52_xp000000/testPlausibility.py -  D 2018_12_01_17_41_08_xp000000/LGneurons.py -  D 2018_12_01_17_41_08_xp000000/__init__.py -  D 2018_12_01_17_41_08_xp000000/filter.py -  D 2018_12_01_17_41_08_xp000000/iniBG.py -  D 2018_12_01_17_41_08_xp000000/modelParams.py -  D 2018_12_01_17_41_08_xp000000/modelParams.pyc -  D 2018_12_01_17_41_08_xp000000/nstrand.py -  D 2018_12_01_17_41_08_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_41_08_xp000000/spikeProcessing.py -  D 2018_12_01_17_41_08_xp000000/testPlausibility.py -  D 2018_12_01_17_41_17_xp000000/LGneurons.py -  D 2018_12_01_17_41_17_xp000000/LGneurons.pyc -  D 2018_12_01_17_41_17_xp000000/__init__.py -  D 2018_12_01_17_41_17_xp000000/filter.py -  D 2018_12_01_17_41_17_xp000000/filter.pyc -  D 2018_12_01_17_41_17_xp000000/iniBG.py -  D 2018_12_01_17_41_17_xp000000/iniBG.pyc -  D 2018_12_01_17_41_17_xp000000/modelParams.py -  D 2018_12_01_17_41_17_xp000000/modelParams.pyc -  D 2018_12_01_17_41_17_xp000000/nstrand.py -  D 2018_12_01_17_41_17_xp000000/nstrand.pyc -  D 2018_12_01_17_41_17_xp000000/solutions_simple_unique.csv -  D 2018_12_01_17_41_17_xp000000/spikeProcessing.py -  D 2018_12_01_17_41_17_xp000000/spikeProcessing.pyc -  D 2018_12_01_17_41_17_xp000000/testPlausibility.py -  D LGneurons.py -  D iniBG.py -  M run.py -  D testPlausibility.py - 

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
    "IeFSI": 0.0,
    "IeGPe": 0.0,
    "IeGPi": 0.0,
    "IeMSN": 0.0,
    "IeProt": 0.0,
    "IeSTN": 0.0,
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