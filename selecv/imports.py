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
    numpy_data = pds.Series(df["Count"]).values.reshape(
        cell_names.size, receptor_names.size
    )  # Rows are in the order of cell_names. Receptor Type is on the order of receptor_names
    numpy_data = numpy_data[:, [1, 2, 4, 0, 3]]  # Rearrange numpy_data to place IL2Ra first, then IL2Rb, then gc, then IL15Ra in this order
    numpy_data = numpy_data[[4, 0, 5, 1, 9, 7, 3, 8, 6, 2], :]  # Reorder to match cells
    return data, numpy_data, cell_names


mutaff = {
    "WT IL-2": [10.0, 144.0],
    "WT N-term": [0.19, 5.296],
    "WT C-term": [0.54, 3.043],
    "V91K C-term": [0.69, 7.5586],
    "R38Q N-term": [0.71, 3.9949],
    "F42Q N-Term": [9.48, 2.815],
    "N88D C-term": [1.01, 24.0166],
}


def getAffDict1():
    """Returns a dictionary containing mutant dissociation constants for 2Ra and BGc"""
    return mutaff


theoreticalpops = {
    r"$R_1^{lo}R_2^{lo}$": [2.2, 2.2, [[0.01, 0.005], [0.005, 0.01]]],
    r"$R_1^{med}R_2^{lo}$": [4, 2.2, [[0.015, 0.00], [0.00, 0.005]]],
    r"$R_1^{hi}R_2^{lo}$": [5.8, 2.2, [[0.015, 0.00], [0.00, 0.005]]],
    r"$R_1^{lo}R_2^{hi}$": [2.2, 5.8, [[0.005, 0.00], [0.00, 0.015]]],
    r"$R_1^{med}R_2^{hi}$": [4.0, 5.6, [[0.01, 0.005], [0.005, 0.01]]],
    r"$R_1^{hi}R_2^{med}$": [5.6, 4.0, [[0.01, 0.005], [0.005, 0.01]]],
    r"$R_1^{hi}R_2^{hi}$": [5.8, 5.8, [[0.01, 0.01], [0.02, 0.01]]],
    r"$R_1^{med}R_2^{med}$": [3.9, 3.9, [[0.05, -0.04], [-0.04, 0.05]]],
}


def getPopDict():
    """Returns dictionary and dataframe containt theoretical populations"""
    populationdf = pds.DataFrame.from_dict(data=theoreticalpops, orient="index", columns=["Receptor_1", "Receptor_2", "Covariance_Matrix"])
    populationdf = populationdf.reset_index()
    populationdf.columns = ["Population", "Receptor_1", "Receptor_2", "Covariance_Matrix"]
    return theoreticalpops, populationdf


affDict = {
    "IL2·IL2Rα": [1 / 10 * 10e8, "Cytokine"],
    "IL2·IL2Rβ": [1 / 144 * 10e8, "Cytokine"],
    "IL15·IL2Rα": [1 / 438 * 10e8, "Cytokine"],
    "IL15·IL15R": [1 / 0.065 * 10e8, "Cytokine"],
    "IL7·IL7R": [1 / 59.0 * 10e8, "Cytokine"],
    "IL9·IL9R": [1 / 0.1 * 10e8, "Cytokine"],
    "IL4·IL4R": [1 / 1.0 * 10e8, "Cytokine"],
    "IL21·IL21R": [1 / 0.07 * 10e8, "Cytokine"],
    "FcGamma": [6.5e7, "FcG"],
}


def getAffDict():
    """Returns dictionary and dataframe containt theoretical populations"""
    affDF = pds.DataFrame.from_dict(affDict, orient="index", columns=["Affinity", "Type"])
    affDF = affDF.reset_index()
    affDF = affDF.rename(columns={"index": "Receptor/Ligand Pair", "Affinity": "Affinity", "Type": "Type"})
    return affDict, affDF
