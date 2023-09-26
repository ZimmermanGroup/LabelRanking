import numpy as np
import pandas as pd
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from abc import ABC, abstractmethod


def yield_to_ranking(yield_array):
    """Transforms an array of yield values to their rankings.
    Currently, treat 0% yields as ties in the last place. (total # of labels)

    Parameters
    ----------
    yield_array : np.ndarray of shape (n_samples, n_conditions)
        Array of raw yield values.

    Returns
    -------
    ranking_array : np.ndarray of shape (n_samples, n_conditions)
        Array of ranking values. Lower values correspond to higher yields.
    """
    if len(yield_array.shape) == 2:
        raw_rank = yield_array.shape[1] - np.argsort(
            np.argsort(yield_array, axis=1), axis=1
        )
        for i, row in enumerate(yield_array):
            raw_rank[i, np.where(row == 0)[0]] = len(row > 0)
        # print("Raw rank", raw_rank)
    elif len(yield_array.shape) == 1:
        raw_rank = len(yield_array) - np.argsort(np.argsort(yield_array))
        raw_rank[np.where(raw_rank == 0)[0]] = len(raw_rank)
    return raw_rank


class Dataset(ABC):
    """
    Base class for preparing datasets.

    Parameters
    ----------
    for_regressor : bool
        Whether the input will be used for training a regressor.
    n_rxns : int
        Number of reactions that we simulate to select and conduct.
    """

    def __init__(self, for_regressor, n_rxns):
        self.for_regressor = for_regressor
        self.n_rxns = n_rxns


class DeoxyDataset:
    """
    Prepares arrays from the deoxyfluorination dataset for downstream use.

    Parameters
    ----------
    for_regressor : bool
        Whether the input will be used for training a regressor.
    component_to_rank : str {'base', 'sulfonyl_fluoride', 'both'}
        Which reaction component to be ranked.
    train_together : bool
        Whether the non-label reaction component should be trained altogether, or used as separate datasets.
        Only considered when component_to_rank is not 'both'.
    n_rxns : int
        Number of reactions that we simulate to conduct.

    Attributes
    ----------
    X_fp : np.2darray of shape (n_samples, n_bits)
    """

    def __init__(self, for_regressor, component_to_rank, train_together, n_rxns):
        self.for_regressor = for_regressor
        self.component_to_rank = component_to_rank
        self.train_together = train_together
        self.n_rxns = n_rxns

        self.substrate_smiles = [
            x[0]
            for x in pd.read_excel(
                "datasets/deoxyfluorination/substrate_smiles.xlsx", header=None
            ).values.tolist()
        ]
        self.descriptors = pd.read_csv(
            "datasets/deoxyfluorination/descriptor_table.csv"
        ).to_numpy()
        self.onehot = pd.read_csv(
            "datasets/deoxyfluorination/descriptor_table-OHE.csv"
        ).to_numpy()
        self.yields = (
            pd.read_csv("datasets/deoxyfluorination/observed_yields.csv", header=None)
            .to_numpy()
            .flatten()
        )
        if self.component_to_rank == "sulfonyl_fluoride":
            self.n_non_rank_component = 4  # 4 bases
        elif self.component_to_rank == "base":
            self.n_non_rank_component = 5  # 5 sulfonyl fluorides

    def _combine_desc_arrays(self, substrate_array, reagent_array, n_base_bits=1):
        """
        Combines feature arrays of a substrate and a reagent that first alters through
        the reagent, followed by substrate.
        e.g. if there are two reagents:
          row1=(substrate1 + reagent1) // row2=(substrate1 + reagent2)
          row3=(substrate2 + reagent1) // row4=(substrate2 + reagent2)

        Parameters
        ---------
        substrate_array : np.ndarray of shape (n_substrates, n_features)
            Feature array of substrates
        reagent_array : np.ndarray of shape (n_reagents, n_features)
            Feature array of reagents
        n_base_bits : int
            Number of descriptor elements that bases take.

        Returns
        -------
        combined_array : np.ndarray of shape (n_substrates * n_reagents, n_features_total)
            Combined feature array.
        """
        if (
            self.component_to_rank == "sulfonyl_fluoride" and not self.for_regressor
        ) or (self.component_to_rank == "base" and self.for_regressor):
            combined_array = np.hstack(
                (
                    np.repeat(substrate_array, 4, axis=0),
                    np.tile(
                        reagent_array[
                            [5 * x for x in range(4)],
                            :n_base_bits,
                        ],
                        (32, 1),
                    ),
                )
            )
        elif (self.component_to_rank == "base" and not self.for_regressor) or (
            self.component_to_rank == "sulfonyl_fluoride" and self.for_regressor
        ):
            combined_array = np.hstack(
                (
                    np.repeat(
                        substrate_array,
                        5,  # number of sulfonyl_fluoride
                        axis=0,
                    ),
                    np.tile(
                        reagent_array[
                            [x for x in range(5)],
                            n_base_bits:,  # adding sulfonyl fluoride features
                        ],
                        (32, 1),
                    ),
                )
            )
        return combined_array

    @property
    def X_fp(self, fpSize=1024, radius=3):
        """
        Prepares fingerprint arrays of substrates.
        For regressors, other descriptors are appended after the substrate fingerprint.

        Parameters
        ----------
        fpSize : int
            Length of the Morgan FP.
        radius : int
            Radius of the Morgan FP.

        Returns
        -------
        X_fp : np.ndarray of shape (n_rxns, n_features)
            n_features depends on self.for_regressor
        """
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
        fp_array = np.zeros((len(self.substrate_smiles), fpSize))
        for i, smiles in enumerate(self.substrate_smiles):
            fp_array[i] = mfpgen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
        reagent_desc = self.descriptors[:, -4:]
        if self.for_regressor:
            if self.component_to_rank == "both" or self.train_together:
                self._X_fp = np.hstack(
                    (
                        np.repeat(fp_array, 20, axis=0),
                        reagent_desc,
                    )
                )
            else:
                self._X_fp = self._combine_desc_arrays(fp_array, reagent_desc)
        else:
            if self.component_to_rank == "both":
                self._X_fp = fp_array
            elif self.component_to_rank in ["sulfonyl_fluoride", "base"]:
                if self.train_together:
                    self._X_fp = self._combine_desc_arrays(fp_array, reagent_desc)
                else:
                    self._X_fp = [fp_array] * self.n_non_rank_component

        return self._X_fp

    @property
    def X_desc(self):
        """
        Prepares descriptor arrays.
        """
        if self.for_regressor:
            if self.component_to_rank == "both" or self.train_together:
                self._X_desc = deepcopy(self.descriptors)
            else:
                self._X_desc = self._combine_desc_arrays(
                    self.descriptors[[20 * x for x in range(32)], :19],
                    self.descriptors[[20 * x for x in range(32)], 19:],
                )
        else:
            if self.component_to_rank == "both":
                self._X_desc = self.descriptors[
                    [20 * x for x in range(32)], :19
                ]  #  19=number of substrate descriptors
            elif self.component_to_rank in ["base", "sulfonyl_fluoride"]:
                if self.train_together:
                    # This case the features of reaction component that is not ranked are inputs
                    self._X_desc = self._combine_desc_arrays(
                        self.descriptors[[20 * x for x in range(32)], :19],
                        self.descriptors[[20 * x for x in range(32)], 19:],
                    )
                else:
                    # This case the substrates are the only inputs.
                    self._X_desc = [
                        self.descriptors[[20 * x for x in range(32)], :19]
                    ] * self.n_non_rank_component

        return self._X_desc

    @property
    def X_onehot(self):
        "Prepares onehot arrays."  # Finished confirming.
        if self.for_regressor:
            if self.component_to_rank == "both" or self.train_together:
                self._X_onehot = deepcopy(self.onehot)
            else:
                self._X_onehot = self._combine_desc_arrays(
                    self.onehot[[20 * x for x in range(32)], :32],
                    self.onehot[:20, 32:],
                    4,
                )
        else:
            if self.component_to_rank == "both":
                self._X_onehot = self.onehot[
                    [20 * x for x in range(32)], :32
                ]  #  19=number of substrate descriptors
            elif self.component_to_rank in ["base", "sulfonyl_fluoride"]:
                if self.train_together:
                    self._X_onehot = self._combine_desc_arrays(
                        self.onehot[[20 * x for x in range(32)], :32],
                        self.onehot[:20, 32:],
                        4,
                    )
                else:
                    self._X_onehot = [
                        self.onehot[[20 * x for x in range(32)], :32]
                    ] * self.n_non_rank_component
        # print(self._X_onehot)
        return self._X_onehot

    def _split_yields(self, y_array, n_rows):
        """If the yield array is a repetition of yields from reactions of a set of reagents,
        it splits the array such that each array is yields from only that reagent.

        Parameters
        ----------
        y_array : np.ndarray of shape (n_substrates*n_rxn_component1, n_rxn_component2)
            Yield array of reactions where rows are repeated by some reaction component.
            i.e. row1 = substrate 1 & reagent A, row2 = substrate 1 & reagent B,
                 row3 = substrate 2 & reagent A, row4 = substrate 2 & reagent B
                 •••
                 with each row having lengths of number of reaction component 2
        n_rows : int
            Number of different reagents used for reaction component 1 as in the description above.

        Returns
        -------
        y_by_reagent : list of n_rxn_component1 number of np.ndarray of shape (n_substrates, n_rxn_component2)
            Yield arrays for each reaction component1.
        """
        y_by_reagent = {}  # [[]] * n_rows
        for i in range(n_rows):
            y_by_reagent.update({i: []})
        for ind, row in enumerate(y_array):
            where = ind % n_rows
            y_by_reagent[where].append(row)
        y_split = []
        for i in range(n_rows):
            y_split.append(np.vstack(tuple(y_by_reagent[i])))
        return y_split  # y_by_reagent

    @property
    def y_yield(self):
        """Prepares continuous yield value arrays appropriate for each problem setting."""
        if self.for_regressor:
            if self.component_to_rank == "both" or self.train_together:
                self._y_yield = deepcopy(self.yields)
            else:
                if self.component_to_rank == "sulfonyl_fluoride":
                    self._y_yield = [
                        x.flatten()
                        for x in self._split_yields(self.yields.reshape(32 * 4, 5), 4)
                    ]
                elif self.component_to_rank == "base":
                    yield_together = np.vstack(
                        tuple(
                            [row.reshape(4, 5).T for row in self.yields.reshape(32, 20)]
                        )
                    )
                    self._y_yield = [
                        x.flatten() for x in self._split_yields(yield_together, 5)
                    ]
        else:
            if self.component_to_rank == "both":
                self._y_yield = self.yields.reshape(32, 20)
            elif self.component_to_rank == "sulfonyl_fluoride":
                if self.train_together:
                    self._y_yield = self.yields.reshape(32 * 4, 5)
                else:
                    self._y_yield = self._split_yields(
                        self.yields.reshape(32 * 4, 5), 4
                    )
            elif self.component_to_rank == "base":
                yield_together = np.vstack(
                    tuple([row.reshape(4, 5).T for row in self.yields.reshape(32, 20)])
                )
                if self.train_together:
                    self._y_yield = deepcopy(yield_together)
                else:
                    self._y_yield = self._split_yields(yield_together, 5)

        return self._y_yield

    @property
    def y_ranking(self):
        """Transforms raw continuous yield values into arrays of ranks."""
        if type(self._y_yield) == list:
            self._y_ranking = [yield_to_ranking(x) for x in self._y_yield]
        else:
            self._y_ranking = yield_to_ranking(self._y_yield)
        return self._y_ranking

    def _transform_yield_to_multilabel(self, yield_array):
        """
        For each substrate (pair), the array of yields under multiple reaction conditions
        are transformed into a multilabel array, marking top-k reaction conditions as positives.

        Parameters
        ----------
        yield_array : np.ndarray of shape (n_substrates, n_reaction_conditions)
            Array of continuous yield values.

        Returns
        -------
        multilabel_array : np.ndarray of shape (n_substrates, n_reaction_conditions)
            Array to be used for multilabel classification.
        """
        positive_inds = list(
            np.argpartition(-1 * yield_array, self.n_rxns, axis=1)[: self.n_rxns]
        )
        multilabel_array = np.zeros_like(yield_array)
        for i, row in enumerate(yield_array):
            min_val_of_positives = min(row[positive_inds])
            multilabel_array[i, np.argwhere(row >= min_val_of_positives)] = 1
        return multilabel_array

    @property
    def y_label(self):
        """
        Prepares output array to be used for conventional classifications.
        If we will select one highest prediction, we formulate a multiclass classification.
        If we select more than one, a multilabel classification is formulated.
        """
        if self.n_rxns > 1:
            if self.train_together:
                self._y_label = self._transform_yield_to_multilabel(self._y_yield)
            else:
                self._y_label = [
                    self._transform_yield_to_multilabel(x) for x in self._y_yield
                ]
        elif self.n_rxns == 1:
            if type(self._y_yield) == list:
                self._y_label = [
                    np.argmin(x, axis=1) for x in np.argmin(self._y_yield, axis=1)
                ]
            else:
                self._y_label = np.argmin(self._y_yield, axis=1)
        return self._y_label
