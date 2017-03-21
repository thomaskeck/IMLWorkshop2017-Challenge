import glob
import root_pandas

for name in ['gluons_modified', 'gluons_standard', 'quarks_modified', 'quarks_standard']:
    print("Convert to pandas", name)
    files = glob.glob(name + "/*.root")
    root_pandas.read_root(files, 'treeJets').to_pickle(name + '.pickle')

