import numpy as np
import pandas as pd
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


DEOXY_SUBSTRATE_SMILES = [
    "OCCCCC1=CC=CC=C1",
    "OC(C)CCC1=CC=CC=C1",
    "OC(C)(C)CCC1=CC=CC=C1",
    "OCCCC1=NC(C2=CC=CC=C2)=C(C3=CC=CC=C3)O1",
    "O=C(C(N(C)C=N1)=C1N2C)N(CCCCC(O)C)C2=O",
    "O=C(C1=CC=CC=C1)N(C(C2=CC=CC=C2)=O)C3=NC=NC4=C3N=CN4[C@H]5[C@H](OC(C6=CC=CC=C6)=O)[C@H](OC(C7=CC=CC=C7)=O)[C@@H](CO)O5",
    "ClC1=CC([N+]([O-])=O)=CC=C1/N=N/C2=CC=C(N(CC)CCO)C=C2",
    "OCC1N(CCCC2=CC=CC=C2)CCCC1",
    "O[C@@H]1C[C@H](OCC2=CC=CC=C2)C1",
    "O=C(OC)[C@H]1N(C(OC(C)(C)C)=O)C[C@H](O)C1",
    "O=C(OC)[C@H]1N(C(OC(C)(C)C)=O)C[C@@H](O)C1",
    "CC1(C)OC[C@H]([C@@H]2[C@@H](O)[C@@H](OC(C)(C)O3)[C@@H]3O2)O1",
    "O[C@@H]1CO[C@]2([H])[C@@H](OC(C)=O)CO[C@@]21[H]",
    "O[C@@H]1CC[C@@H](N2C(C(C=CC=C3)=C3C2=O)=O)CC1",
    "O[C@H]1CC[C@@H](N2C(C(C=CC=C3)=C3C2=O)=O)CC1",
    "O[C@]1([H])C[C@H](CC2)N(C(OC(C)(C)C)=O)[C@H]2C1",
    "OCC(C=C1)=CC=C1C2=CC=CC=C2",
    "OCC1=CC=C(S(N(CCC)CCC)(=O)=O)C=C1",
    "ClC(N=C1CCCC)=C(CO)N1CC(C=C2)=CC=C2C3=C(C4=NN=NN4C(C5=CC=CC=C5)(C6=CC=CC=C6)C7=CC=CC=C7)C=CC=C3",
    "OCN1C(C)=CC(C)=N1",
    "COC1=C(OCCCC(OC)=O)C=C([N+]([O-])=O)C(C(O)C)=C1",
    "O[C@@H](C1=CC=CC=C1)[C@H](C)N2CCCC2",
    "C/C(C)=C/CC/C(C)=C/CC/C(C)=C/CO",
    "CCCCCCCC(O)C=C",
    "C/C(C)=C/CCC(O)(C=C)C",
    "O=C(C1=CC=C(Cl)C=C1)N2C3=CC=C(OC)C=C3C(CCO)=C2C",
    "CC1=NC=C([N+]([O-])=O)N1CCO",
    "O[C@@H](C1)CC[C@@]2(C)C1=CC[C@]3([H])[C@]2([H])CC[C@@]4(C)[C@@]3([H])CC=C4C5=CN=CC=C5",
    "O=C1C(O)CCO1",
    "O=C1C=C[C@@]2(C)C(CC[C@]([C@@](CC[C@@]3(C(CO)=O)O)([H])[C@]3(C)C4)([H])[C@]2([H])C4=O)=C1",
    "OC[C@@H](C(OC)=O)NC(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
    "C[C@@]12CC[C@]3([H])[C@]4(OO2)[C@@](O[C@H](O)[C@H](C)C4CC[C@H]3C)([H])O1",
]


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


class DeoxyDataset :
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
    def __init__(
        self,
        for_regressor,
        component_to_rank,
        train_together,
        n_rxns
    ) :
        self.for_regressor = for_regressor
        self.component_to_rank = component_to_rank
        self.train_together = train_together
        self.n_rxns = n_rxns

        self.substrate_smiles = DEOXY_SUBSTRATE_SMILES
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
    
    def _combine_desc_arrays(self, substrate_array, reagent_array, n_base_bits=1):
        if self.component_to_rank == "sulfonyl_fluoride" :
            combined_array = np.hstack((
                np.repeat(substrate_array, 4, axis=0),
                np.tile(
                    reagent_array[
                        [5 * x for x in range(4)],
                        :n_base_bits,
                    ],
                    (32, 1),
                )
            ))
        elif self.component_to_rank == "base":
            combined_array = np.hstack((
                np.repeat(
                    substrate_array,
                    5, # number of sulfonyl_fluoride
                    axis=0,
                ),  
                np.tile(
                    reagent_array[
                        [x for x in range(5)], n_base_bits:
                    ],
                    (32, 1),
                ),
            ))
        return combined_array


    @property
    def X_fp(self, fpSize=1024, radius=3) :
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
        if self.for_regressor :
            self._X_fp = np.hstack((
                np.repeat(fp_array, 20, axis=0),
                reagent_desc,
            ))
        else :
            if self.component_to_rank == "both" :
                self._X_fp = fp_array
            elif self.component_to_rank in ["sulfonyl_fluoride", "base"]:
                if self.train_together :
                    self._X_fp = self._combine_desc_arrays(fp_array, reagent_desc)
                if not self.train_together :
                    if self.component_to_rank == "sulfonyl_fluoride" :
                        self._X_fp = [fp_array]*4
                    else :
                        self._X_fp = [fp_array]*5
        return self._X_fp
    
    
    @property
    def X_desc(self) : 
        """ 
        Prepares descriptor arrays.
        """
        if self.for_regressor :
            self._X_desc = deepcopy(self.descriptors)
        else :
            if self.component_to_rank == "both":
                self._X_desc = self.descriptors[[20 * x for x in range(32)], :19] #  19=number of substrate descriptors
            elif self.component_to_rank in ["base", "sulfonyl_fluoride"] :
                if self.train_together :
                    self._X_desc = self._combine_desc_arrays(
                        self.descriptors[[20 * x for x in range(32)], :19],
                        self.descriptors[[20 * x for x in range(32)], 19:]
                    )
                else :
                    if self.component_to_rank == "sulfonyl_fluoride" :
                        self._X_desc = [self.descriptors[[20 * x for x in range(32)], :19]]*4
                    else :
                        self._X_desc = [self.descriptors[[20 * x for x in range(32)], :19]]*5
        return self._X_desc

    @property
    def X_onehot(self):
        " Prepares onehot arrays."
        if self.for_regressor :
            self._X_onehot = deepcopy(self.onehot)
        else :
            if self.component_to_rank == "both" :
                self._X_onehot = self.onehot[[20 * x for x in range(32)], :32] #  19=number of substrate descriptors
            elif self.component_to_rank in ["base", "sulfonyl_fluoride"] :
                if self.train_together :
                    self._X_onehot = self._combine_desc_arrays(
                        self.onehot[[20 * x for x in range(32)], :32],
                        self.onehot[[20 * x for x in range(32)], 32:],
                        4
                    )
                else :
                    if self.component_to_rank == "sulfonyl_fluoride" :
                        self._X_onehot = [self.onehot[[20 * x for x in range(32)], :32]]*4
                    else :
                        self._X_onehot = [self.onehot[[20 * x for x in range(32)], :32]]*5
        return self._X_onehot
    

    def _split_yields(self, y_array, n_rows):
        y_by_reagent = [[]] * n_rows
        for ind, row in enumerate(y_array) :
            where = ind % n_rows
            y_by_reagent[where].append(row)
        return y_by_reagent

    @property
    def y_yield(self):
        if self.for_regressor : 
            self._y_yield = deepcopy(self.yields)
        else :
            if self.component_to_rank == "both" :
                self._y_yield = self.yields.reshape(32,20)
            elif self.component_to_rank == "sulfonyl_fluoride" :
                if self.train_together :
                    self._y_yield = self.yields.reshape(32 * 4, 5)
                else :
                    yield_by_reagent = self._split_yields(self.yields.reshape(32 * 4, 5), 4)
                    self._y_yield = [np.vstack(tuple(x)) for x in yield_by_reagent]
            elif self.component_to_rank == "base":
                yield_together = np.vstack(tuple([row.reshape(4,5).T for row in self.yields.reshape(32,20)]))
                if self.train_together :
                    self._y_yield = deepcopy(yield_together)
                else : 
                    yield_by_reagent = self._split_yields(yield_together, 5)
                    self._y_yield = [np.vstack(tuple(x)) for x in yield_by_reagent]

        return self._y_yield

    @property
    def y_ranking(self):
        if type(self._y_yield) == list :
            self._y_ranking = [yield_to_ranking(x) for x in self._y_yield]
        else :
            self._y_ranking = yield_to_ranking(self._y_yield)
        # if self.component_to_rank == "both":
        #     self._y_ranking = yield_to_ranking(self.yields.reshape(32, 20))
        # elif self.component_to_rank == "sulfony_fluoride" :
        #     self._y_ranking = yield_to_ranking(self.yields.reshape(32 * 4, 5))
        # elif self.component_to_rank == "base":
        #     self._y_ranking = yield_to_ranking(self.yields.reshape(32 * 5, 4))
        return self._y_ranking
    

    def _transform_to_multilabel(self, yield_array) :
        positive_inds = list(np.argpartition(-1*yield_array, self.n_rxns, axis=1)[:self.n_rxns])
        multilabel_array = np.zeros_like(yield_array)
        for i, row in enumerate(yield_array):
            min_val_of_positives = min(row[positive_inds])
            multilabel_array[i, np.argwhere(row >= min_val_of_positives)] = 1
        return multilabel_array

    @property
    def y_label(self):
        """ 
        Prepares output array to be used for conventional classifications.
        """
        if self.n_rxns > 1:
            if self.train_together :
                self._y_label = self._transform_to_multilabel(self._y_yield)
            else : 
                self._y_label = [self._transform_to_multilabel(x) for x in self._y_yield]
        elif self.n_rxns == 1 :
            if type(self._y_yield) == list:
                self._y_label = [np.argmin(x, axis=1) for x in np.argmin(self._y_yield, axis=1)]
            else : 
                self._y_label = np.argmin(self._y_yield, axis=1)
        return self._y_label
        