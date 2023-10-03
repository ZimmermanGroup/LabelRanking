import numpy as np
import pandas as pd
from copy import deepcopy
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from abc import ABC
import joblib


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

    @property
    def X_dist(self):
        """Tanimoto distances between the substrates, used for neighbor based models."""
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
        cfp_nonnp = [
            mfpgen.GetCountFingerprint(Chem.MolFromSmiles(x)) for x in self.smiles_list
        ]
        dists = np.zeros((len(cfp_nonnp), len(cfp_nonnp)))
        for i in range(1, len(cfp_nonnp)):
            similarities = DataStructs.BulkTanimotoSimilarity(
                cfp_nonnp[i], cfp_nonnp[:i]
            )
            dists[i, :i] = np.array([1 - x for x in similarities])
        dists += dists.T
        self._X_dist = dists
        # print("DISTARRAY",dists)
        return self._X_dist


class NatureDataset(Dataset):
    """
    Prepares arrays from the HTE paper in Nature, 2018.

    Parameters
    ----------
    for_regressor : bool
        Whether the input will be used for training a regressor.
    n_rxns : int
        Number of reactions that we simulate to select and conduct.
    component_to_rank : str {'amine', 'amide', 'sulfonamide'}
        Which substrate dataset type to use.
        Although 'substrate_to_rank' is a better name, using this for consistency.
    """

    def __init__(self, for_regressor, component_to_rank, n_rxns=1):
        super().__init__(for_regressor, n_rxns)
        self.component_to_rank = component_to_rank

        # Loading the raw dataset
        raw_data = pd.read_excel(
            "datasets/natureHTE/natureHTE.xlsx",
            sheet_name="Report - Variable Conditions",
            usecols=["BB SMILES", "Chemistry", "Catalyst", "Base", "Rel. % Conv."],
        )
        # Reagent descriptors
        base_descriptors = pd.read_excel(
            "datasets/natureHTE/reagent_desc.xlsx", sheet_name="Base"
        )
        cat_descriptors = pd.read_excel(
            "datasets/natureHTE/reagent_desc.xlsx", sheet_name="Catalyst"
        )
        self.reagent_data = {}
        for _, row in base_descriptors.iterrows():
            self.reagent_data.update({row[0]: row[1:].to_numpy()})
        for _, row in cat_descriptors.iterrows():
            self.reagent_data.update({row[0]: row[1:].to_numpy()})
        # Reaction data by substrate type
        if self.component_to_rank == "amine":
            self.df = raw_data[raw_data["Chemistry"] == "Amine"]
            self.validation_rows = joblib.load(
                "datasets/natureHTE/nature_amine_inds_to_remove.joblib"
            )
        elif self.component_to_rank == "sulfonamide":
            self.df = raw_data[raw_data["Chemistry"] == "Sulfonamide"].reset_index()
            self.validation_rows = joblib.load(
                "datasets/natureHTE/nature_sulfon_inds_to_remove.joblib"
            )
        elif self.component_to_rank == "amide":
            self.df = raw_data[raw_data["Chemistry"] == "Amide"].reset_index()
            self.validation_rows = joblib.load(
                "datasets/natureHTE/nature_amide_inds_to_remove.joblib"
            )
        self.smiles_list = self.df["BB SMILES"].unique().tolist()
        self.cats = self.df["Catalyst"].unique().tolist()
        self.bases = self.df["Base"].unique().tolist()
        self.n_rank_component = len(self.cats) * len(self.bases)
        self.train_together = False # for kNN

    def _split_train_validation(self, array):
        """Splits prepared X or Y array into a training set with more balanced output.

        Parameters
        ----------
        array : np.ndarray of shape (n_rxns or n_substrates, n_features) or (n_rxns or n_substrates,)
            Array to split.
        """
        if self.for_regressor:
            train_rows = []
            val_rows = []
            for i in range(len(self.df["BB SMILES"].unique())):
                if i not in self.validation_rows:
                    train_rows.extend([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3])
                if i in self.validation_rows:
                    val_rows.extend([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3])
            array_train = array[train_rows]
            array_val = array[val_rows]
        else:
            array_train = array[
                [x for x in range(array.shape[0]) if x not in self.validation_rows]
            ]
            array_val = array[self.validation_rows]
        return array_train, array_val

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
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
        if self.for_regressor:
            fp_arrays = []
            for i, row in self.df.iterrows():
                fp_array = mfpgen.GetCountFingerprintAsNumPy(
                    Chem.MolFromSmiles(row["BB SMILES"])
                ).reshape(1, -1)
                cat_desc = self.reagent_data[row["Catalyst"]].reshape(1, -1)
                base_desc = self.reagent_data[row["Base"]].reshape(1, -1)
                fp_arrays.append(np.hstack((fp_array, cat_desc, base_desc)))
        else:
            fp_arrays = [
                mfpgen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(x))
                for x in self.df["BB SMILES"].unique()
            ]
        self._X_fp, self.X_valid = self._split_train_validation(np.vstack(tuple(fp_arrays)))
        return self._X_fp

    @property
    def X_onehot(self):
        """Prepares onehot arrays."""
        n_subs = len(self.smiles_list)
        if self.for_regressor:
            n_cats = len(self.cats)
            n_bases = len(self.bases)
            array = np.zeros((self.df.shape[0], n_subs + n_cats + n_bases))
            for i, row in self.df.iterrows():
                array[
                    [i, i, i],
                    [
                        self.smiles_list.index(row["BB SMILES"]),
                        n_subs + self.cats.index(row["Catalyst"]),
                        n_subs + n_bases + self.bases.index(row["Base"]),
                    ],
                ] = 1
        else:
            array = np.identity(n_subs)
        self._X_onehot, self.X_valid = self._split_train_validation(array)
        return self._X_onehot

    @property
    def X_dist(self):
        """Tanimoto distances between the substrates, used for neighbor based models."""
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
        train_cfp = [mfpgen.GetCountFingerprint(Chem.MolFromSmiles(x)) for i, x in enumerate(self.smiles_list) if i not in self.validation_rows]
        valid_cfp = [mfpgen.GetCountFingerprint(Chem.MolFromSmiles(x)) for i, x in enumerate(self.smiles_list) if i in self.validation_rows]
        train_dists = np.zeros((len(train_cfp), len(train_cfp)))
        valid_dists = np.zeros((len(valid_cfp), len(train_cfp)))
        for i in range(1, len(train_dists)):
            similarities_btw_train = DataStructs.BulkTanimotoSimilarity(
                train_cfp[i], train_cfp[:i]
            )
            test_to_train_sim = DataStructs.BulkTanimotoSimilarity(
                train_cfp[i], valid_cfp
            )
            train_dists[i, :i] = np.array([1 - x for x in similarities_btw_train])
            valid_dists[:, i] = np.array([1-x for x in test_to_train_sim])
        train_dists += train_dists.T
        self._X_dist = train_dists
        self.X_valid = valid_dists
        return self._X_dist
                
    @property
    def y_yield(self):
        if self.for_regressor:
            y = self.df["Rel. % Conv."].values.flatten()
            y[np.argwhere(np.isnan(y))] = 0
            self._y_yield, self.y_valid = self._split_train_validation(y)
        else:
            self._y_yield, self.y_valid = self._split_train_validation(self._sort_yield_by_substrate())
        return self._y_yield

    def _sort_yield_by_substrate(self):
        """ Prepares an array of yields where each row corresponds to a substrate and column corresponds to reactions conditions. """
        array = np.zeros((len(self.smiles_list), len(self.cats) * len(self.bases)))
        for i, row in self.df.iterrows():
            y_val = row["Rel. % Conv."]
            if np.isnan(y_val):
                y_val = 0
            array[
                self.smiles_list.index(row["BB SMILES"]),
                len(self.cats) * self.cats.index(row["Catalyst"])
                + self.bases.index(row["Base"]),
            ] = y_val
        return array

    @property
    def y_ranking(self):
        self._y_ranking, self.y_valid = self._split_train_validation(
            yield_to_ranking(self._sort_yield_by_substrate())
        )
        return self._y_ranking

    @property
    def y_label(self):
        self._y_label, self.y_valid = self._split_train_validation(
            np.argmax(self._sort_yield_by_substrate(), axis=1)
        )
        return self._y_label
    

class InformerDataset(Dataset):
    """
    Prepares arrays from the nickella-photoredox informer for downstream use.

    Parameters
    ----------
    for_regressor : bool
        Whether the input will be used for training a regressor.
    n_rxns : int
        Number of reactions that we simulate to select and conduct.
    train_together : bool
        Whether the non-label reaction component should be trained altogether, or used as separate datasets.
        Only considered when component_to_rank is not 'both'.
    component_to_rank : str {'both', 'amine_ratio', 'catalyst_ratio'}
        Which reaction component to be ranked.

    Attributes
    ----------
    X_fp :
        Only fingerprint arrays are considered for substrates.
    """

    def __init__(self, for_regressor, component_to_rank, train_together, n_rxns):
        super().__init__(for_regressor, n_rxns)
        self.component_to_rank = component_to_rank
        self.train_together = train_together

        # Reading in the raw dataset
        informer_df = pd.read_excel("datasets/Informer.xlsx").iloc[:40, :]
        desc_df = pd.read_excel(
            "datasets/Informer.xlsx", sheet_name="descriptors", usecols=[0, 1, 2, 3, 4]
        ).iloc[:40, :]
        smiles = pd.read_excel(
            "datasets/Informer.xlsx", sheet_name="smiles", header=None
        )

        # Dropping compounds where all yields are below 20%
        cols_to_erase = []
        for col in informer_df.columns:
            if np.all(informer_df.loc[:, col].to_numpy() < 20):
                cols_to_erase.append(col)
        informer_df = informer_df.loc[
            :, [x for x in range(1, 19) if x not in cols_to_erase]
        ]  # leaves 11 compounds
        smiles_list = [
            x[0]
            for i, x in enumerate(smiles.values.tolist())
            if i + 1 not in cols_to_erase
        ]

        # Assigning the arrays
        self.df = informer_df
        self.desc = desc_df.to_numpy()
        self.smiles_list = smiles_list

        if self.component_to_rank == "amine_ratio":
            self.n_rank_component = 10
            self.n_non_rank_component = 4  # 4 catalyst ratio values
        elif self.component_to_rank == "catalyst_ratio":
            self.n_rank_component = 20
            self.n_non_rank_component = 2  # 2 amine ratio values

    def _split_arrays(self, substrate_array_to_process, other_array, return_X=True):
        if self.for_regressor:
            arrays = []
            for i, (_, yield_vals) in enumerate(self.df.items()):
                if return_X:
                    arrays.append(
                        np.hstack(
                            (
                                np.tile(substrate_array_to_process[i, :], (40, 1)),
                                other_array,
                            )
                        )
                    )
                else:
                    arrays.append(yield_vals.to_numpy())
            if return_X:
                array = np.vstack(tuple(arrays))
            else:
                array = np.concatenate(tuple(arrays))
            if self.train_together:
                processed_array = array
            else:
                if self.component_to_rank == "catalyst_ratio":
                    processed_array = [
                        array[[x for x in range(array.shape[0]) if x % 8 < 4]],
                        array[[x for x in range(array.shape[0]) if x % 8 >= 4]],
                    ]
                elif self.component_to_rank == "amine_ratio":
                    processed_array = [
                        array[[y for y in range(array.shape[0]) if y % 4 == x]]
                        for x in range(4)
                    ]
                if return_X:
                    assert np.all(processed_array[0] == processed_array[1])
                    processed_array = processed_array[0]
        else:
            yield_array = self.df.to_numpy()
            if self.component_to_rank == "amine_ratio":
                if not return_X:
                    if not self.train_together:
                        processed_array = [
                            yield_array[
                                [y for y in range(yield_array.shape[0]) if y % 4 == x],
                                :,
                            ].T
                            for x in range(4)  # 11 x 10
                        ]
                    else:
                        first_amine = []
                        for i, row in enumerate(yield_array.T):
                            first_amine.append([])
                            for j in range(10):  # 10 chunks of 4-catalyst ratio values
                                first_amine[i].append(
                                    row[4 * j : 4 * (j + 1)].reshape(-1, 1)
                                )
                        processed_array = np.vstack(
                            tuple(
                                [
                                    np.hstack(tuple(sub_array))
                                    for sub_array in first_amine
                                ]
                            )
                        )

                else:
                    if not self.train_together:
                        processed_array = [substrate_array_to_process] * 4
                    else:
                        processed_array = np.hstack(
                            (
                                np.repeat(substrate_array_to_process, 4, axis=0),
                                other_array,
                            )
                        )
            elif self.component_to_rank == "catalyst_ratio":
                if not return_X:
                    if not self.train_together:
                        processed_array = [
                            yield_array[
                                [x for x in range(yield_array.shape[0]) if x % 8 < 4], :
                            ].T,  # 11 x 20
                            yield_array[
                                [x for x in range(yield_array.shape[0]) if x % 8 >= 4],
                                :,
                            ].T,
                        ]
                    else:
                        if not self.train_together:
                            processed_array = yield_array.reshape()
                        else:
                            first_amine = []
                            second_amine = []
                            for i, row in enumerate(yield_array.T):
                                first_amine.append([])
                                second_amine.append([])
                                for j in range(
                                    10
                                ):  # 10 chunks of 4-catalyst ratio values
                                    if j % 2 == 0:
                                        first_amine[i].append(
                                            row[4 * j : 4 * (j + 1)].reshape(1, -1)
                                        )
                                    else:
                                        second_amine[i].append(
                                            row[4 * j : 4 * (j + 1)].reshape(1, -1)
                                        )
                            subs_arrays = []
                            for row_first_amine, row_second_amine in zip(
                                first_amine, second_amine
                            ):
                                subs_arrays.append(
                                    np.vstack(
                                        (
                                            np.hstack(tuple(row_first_amine)),
                                            np.hstack(tuple(row_second_amine)),
                                        )
                                    )
                                )
                            processed_array = np.vstack(tuple(subs_arrays))
                else:
                    if not self.train_together:
                        processed_array = [substrate_array_to_process] * 2
                    else:
                        processed_array = np.hstack(
                            (
                                np.repeat(substrate_array_to_process, 2, axis=0),
                                other_array,
                            )
                        )

        return processed_array

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
        cfp = [
            mfpgen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(x))
            for x in self.smiles_list
        ]
        cfp_array = np.vstack(tuple(cfp))
        if self.for_regressor:
            if self.train_together:
                other_array = self.desc
            elif self.component_to_rank == "amine_ratio":
                other_array = np.hstack(
                    (self.desc[:, :3], self.desc[:, -1].reshape(-1, 1))
                )
            elif self.component_to_rank == "catalyst_ratio":
                other_array = self.desc[:, :4]
        else:
            if not self.train_together:
                other_array = None
            else:
                if self.component_to_rank == "amine_ratio":
                    other_array = np.repeat(
                        self.desc[:4, :-1], 11, axis=0
                    )  # 20  match 44
                elif self.component_to_rank == "catalyst_ratio":
                    other_array = np.repeat(
                        np.vstack(
                            tuple(
                                [
                                    row
                                    for i, row in enumerate(
                                        np.hstack(
                                            (
                                                self.desc[:8, :3],
                                                self.desc[:8, -1].reshape(-1, 1),
                                            )
                                        )
                                    )
                                    if i % 4 == 0
                                ]
                            )
                        ),
                        11,
                        axis=0,
                    )  # 10 match 22
        self._X_fp = self._split_arrays(cfp_array, other_array)
        return self._X_fp

    @property
    def X_desc(self, fpSize=1024, radius=3):
        return self.X_fp(fpSize=fpSize, radius=radius)

    @property
    def X_onehot(self):
        "Prepares onehot arrays."
        substrate_onehot_array = np.identity(len(self.smiles_list))
        photocat_onehot_array = np.repeat(np.identity(5), 8, axis=0)
        cat_ratio_onehot_array = np.tile(np.identity(4), (10, 1))
        amine_ratio_onehot_array = np.tile(np.repeat(np.identity(2), 4, axis=0), (5, 1))
        self._X_onehot = self._split_arrays(
            substrate_onehot_array,
            np.hstack(
                (
                    photocat_onehot_array,
                    cat_ratio_onehot_array,
                    amine_ratio_onehot_array,
                )
            ),
        )
        return self._X_onehot

    @property
    def y_yield(self):
        self._y_yield = self._split_arrays(None, None, return_X=False)
        print(len(self._y_yield))
        return self._y_yield

    @property
    def y_ranking(self):
        if type(self.y_yield) == list:
            self._y_ranking = [yield_to_ranking(x) for x in self.y_yield]
        elif type(self.y_yield) == np.ndarray:
            self._y_ranking = yield_to_ranking(self.y_yield)
        return self._y_ranking

    @property
    def y_label(self):
        yields = self.y_yield
        if type(yields) == list:
            labels = []
            for i, y in enumerate(yields):
                label = np.zeros_like(y)
                # print(y)
                nth_highest_yield = np.partition(y, -1 * self.n_rxns, axis=1)[
                    :, -1 * self.n_rxns
                ]
                label[
                    y
                    >= np.hstack(tuple([nth_highest_yield.reshape(-1, 1)] * y.shape[1]))
                ] = 1
                # print(label)
                assert np.all(np.sum(label, axis=1) >= self.n_rxns)
                labels.append(label)
        elif type(yields) == np.ndarray:
            labels = np.zeros_like(yields)
            # print(yields)
            nth_highest_yield = np.partition(yields, -1 * self.n_rxns, axis=1)[
                :, -1 * self.n_rxns
            ]
            labels[
                yields
                >= np.hstack(
                    tuple([nth_highest_yield.reshape(-1, 1)] * yields.shape[1])
                )
            ] = 1
            # print(labels)
            # print(len(np.sum(labels, axis=1)))
            assert np.all(np.sum(labels, axis=1) >= self.n_rxns)
        self._y_label = labels
        return self._y_label


class DeoxyDataset(Dataset):
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

        self.smiles_list = [
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
            self.n_rank_component = 5
            self.n_non_rank_component = 4  # 4 bases
        elif self.component_to_rank == "base":
            self.n_rank_component = 4
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
        fp_array = np.zeros((len(self.smiles_list), fpSize))
        for i, smiles in enumerate(self.smiles_list):
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
        if type(self.y_yield) == list:
            self._y_ranking = [yield_to_ranking(x) for x in self.y_yield]
        else:
            self._y_ranking = yield_to_ranking(self.y_yield)
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
