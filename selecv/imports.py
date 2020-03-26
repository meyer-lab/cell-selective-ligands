"""
Data and file imports.
"""
from os.path import join, dirname
from scipy.stats import gmean
import pandas as pds


path_here = dirname(dirname(__file__))


def import_Rexpr():
    """ Loads CSV file containing Rexpr levels from Visterra data. """
    data = pds.read_csv(join(path_here, "selecv/data/final_receptor_levels.csv"))  # Every row in the data represents a specific cell
    df = data.groupby(["Cell Type", "Receptor"]).agg(gmean)  # Get the mean receptor count for each cell across trials in a new dataframe.
    cell_names, receptor_names = df.index.unique().levels  # IL15Ra_idx=0|IL2Ra_idx=1|IL2Rb_idx=2|IL7Ra_idx=3|gc_idx=4|
    cell_names = cell_names[[4, 0, 5, 1, 9, 7, 3, 8, 6, 2]]  # Reorder to match pstat import order
    receptor_names = receptor_names[[1, 2, 4, 0, 3]]  # Reorder so that IL2Ra_idx=0|IL2Rb_idx=1|gc_idx=2|IL15Ra_idx=3|IL7Ra_idx=4
    numpy_data = pds.Series(df["Count"]).values.reshape(cell_names.size, receptor_names.size)  # Rows are in the order of cell_names. Receptor Type is on the order of receptor_names
    numpy_data = numpy_data[:, [1, 2, 4, 0, 3]]  # Rearrange numpy_data to place IL2Ra first, then IL2Rb, then gc, then IL15Ra in this order
    numpy_data = numpy_data[[4, 0, 5, 1, 9, 7, 3, 8, 6, 2], :]  # Reorder to match cells
    return data, numpy_data, cell_names


mutaff = {
    'WT IL-2': [10.0, 144.0],
    'WT N-term': [0.19, 5.296],
    'WT C-term': [0.54, 3.043],
    'V91K C-term': [0.69, 7.5586],
    'R38Q N-term': [0.71, 3.9949],
    'F42Q N-Term': [9.48, 2.815],
    'N88D C-term': [1.01, 24.0166]
}


def getAffDict():
    """Returns a dictionary containing mutant dissociation constants for 2Ra and BGc"""
    return mutaff


theoreticalpops = {
    'Pop1': [2, 2, [[0.01, 0.005], [0.005, 0.01]]],
    'Pop2': [3, 2, [[0.015, 0.00], [0.00, 0.005]]],
    'Pop3': [4, 2, [[0.015, 0.00], [0.00, 0.005]]],
    'Pop4': [2, 4, [[0.005, 0.00], [0.00, 0.015]]],
    'Pop5': [3.2, 3.8, [[0.01, 0.005], [0.005, 0.01]]],
    'Pop6': [3.8, 3.2, [[0.01, 0.005], [0.005, 0.01]]],
    'Pop7': [4, 4, [[0.01, 0.01], [0.02, 0.01]]]
}


def getPopDict():
    """Returns dictionary and dataframe containt theoretical populations"""
    populationdf = pds.DataFrame.from_dict(data=theoreticalpops, orient='index', columns=['Receptor_1', 'Receptor_2', 'Covariance_Matrix'])
    populationdf = populationdf.reset_index()
    populationdf.columns = ['Population', 'Receptor_1', 'Receptor_2', 'Covariance_Matrix']
    return theoreticalpops, populationdf


affDict = {
    'IL2Rα': [1 / 10 * 10e8, 'cytokine'],
    'IL-2Rβ': [1 / 144 * 10e8, 'cytokine'],
    'FcGamma': [6.5e7, 'FcG']
}


def getAffDict():
    """Returns dictionary and dataframe containt theoretical populations"""
    affDF = pds.DataFrame.from_dict(affDict, orient='index', columns=['Affinity', 'Type'])
    affDF = affDF.reset_index()
    affDF = affDF.rename(columns={'index': 'Ligand', 'Affinity': 'Affinity', 'Type': 'Type'})
    return affDict, affDF
