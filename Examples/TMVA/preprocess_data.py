#!/usr/bin/env python
# Author: Stefan Wunsch, 2017


'''
Example preprocessing of gluons and quark jets data
from IML challenge
'''


def process_arguments():
    import argparse

    '''
    Process input arguments

    Returns:
        args: command-line arguments
    '''
    parser = argparse.ArgumentParser(description='Preprocess data for IML'+\
            'challenge. We read in the data and apply zero-padding on the'+\
            'dynamic value of tracks and tower. In detail, this means, e.g.,'+\
            'that if we fix the number of tracks to 5 but there are only 3'+\
            'tracks, the other tracks variables are filled with zeros.')
    parser.add_argument('--gluons', '-g', type=str, default='',
            help='ROOT files with gluon jets')
    parser.add_argument('--quarks', '-q', type=str, default='',
            help='ROOT files with quark jets')
    parser.add_argument('--tracks', '-tr', type=int, default=5,
            help='Number of towers added to the output file')
    parser.add_argument('--towers', '-to', type=int, default=5,
            help='Number of towers added to the output file')
    parser.add_argument('--output', '-o', type=str, default='preprocessed_data.root',
            help='Output ROOT file with combined data from input files')

    args = parser.parse_args()
    if args.gluons == '':
        raise Exception('Please specify ROOT files for option --gluons')
    if args.quarks == '':
        raise Exception('Please specify ROOT files for option --quarks')

    return args


def combine_files(filenames, tree):
    '''
    Take filenames of ROOT files and combine them in a TChain

    Args:
        filenames: String with filenames separated by blanks
        tree: Name of tree in files

    Returns:
        chain: TChain that holds data of all input ROOT files
    '''

    chain = ROOT.TChain(tree)
    for filename in filenames.strip().split(' '):
        chain.AddFile(filename)

    return chain


def add_tree(output_file, tree_name, chain, num_towers, num_tracks):
    '''
    Add tree to output ROOT file with zero-padded events from chain

    Args:
        output_file: Output ROOT file
        tree_name: Name of created tree
        chain: TChain with events
        num_towers: Number of tower that should be kept or zero-padded
        num_tracks: Number of tracks that should be kept or zero-padded
    '''

    from array import array

    tree = ROOT.TTree(tree_name, tree_name)

    # Define variables
    variables = {}
    single_variables = ['jetPt', 'jetEta', 'jetPhi', 'jetMass', 'ntracks', 'ntowers']
    for name in single_variables:
        variables[name] = array('f', [-999])
        tree.Branch(name, variables[name], '{0}/F'.format(name))

    track_variables = ['trackPt', 'trackEta', 'trackPhi', 'trackCharge']
    for name in track_variables:
        for i in range(num_tracks):
            name_ = '{0}_{1}'.format(name, i)
            variables[name_] = array('f', [-999])
            tree.Branch(name_, variables[name_], '{0}/F'.format(name_))

    tower_variables = ['towerE', 'towerEem', 'towerEhad', 'towerEta', 'towerPhi']
    for name in tower_variables:
        for i in range(num_towers):
            name_ = '{0}_{1}'.format(name, i)
            variables[name_] = array('f', [-999])
            tree.Branch(name_, variables[name_], '{0}/F'.format(name_))

    # Load events from chain and push to output file
    for event in chain:
        # Add single variables
        for name in single_variables:
            variables[name][0] = getattr(event, name)

        # Add track variables
        for name in track_variables:
            values = getattr(event, name)
            # Set all variables to zero
            for i in range(num_tracks):
                variables['{0}_{1}'.format(name, i)][0] = 0.0
            # Set variables and potentially leave the zeros untouched
            for i in range(min(num_tracks, len(values))):
                variables['{0}_{1}'.format(name, i)][0] = values[i]

        # Add tower variables
        for name in tower_variables:
            values = getattr(event, name)
            # Set all variables to zero
            for i in range(num_towers):
                variables['{0}_{1}'.format(name, i)][0] = 0.0
            # Set variables and potentially leave the zeros untouched
            for i in range(min(num_towers, len(values))):
                variables['{0}_{1}'.format(name, i)][0] = values[i]

        tree.Fill()

    # Write tree to file
    tree.Write()


if __name__=='__main__':
    # Get arguments
    args = process_arguments()

    # Combine input ROOT files
    import ROOT # import this here so that it does not overwrite our argparse config
    chain_gluons = combine_files(args.gluons, 'treeJets')
    chain_quarks = combine_files(args.quarks, 'treeJets')

    print('Loaded {0} quark jet events from {1} files.'.format(
            chain_quarks.GetEntries(), len(args.quarks.strip().split(' '))))
    print('Loaded {0} gluon jet events from {1} files.'.format(
            chain_gluons.GetEntries(), len(args.gluons.strip().split(' '))))

    # Create output ROOT file
    output = ROOT.TFile(args.output, 'RECREATE')

    # Do zero-padding and add trees to output ROOT file
    add_tree(output, 'gluons', chain_gluons, args.towers, args.tracks)
    add_tree(output, 'quarks', chain_quarks, args.towers, args.tracks)

    # Write file and close
    output.Write()
    output.Close()
