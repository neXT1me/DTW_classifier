import numpy as np
from tslearn.metrics import dtw_path, dtw

class DTW_classification():
    def __init__(self, train_true, test_fake, test_true=None):
        self.train_true = train_true
        self.test_fake = test_fake
        self.test_true = test_true

        self.value_sign = len(self.train_true)
        self.value_param = len(self.train_true[0])

        self.id_sign = list(range(self.value_sign))
        self.mass_params = [[data[i] for data in self.train_true] for i in range(self.value_param)]

        self.g_ref = []
        self._g_ref()

        self.summ_dtw_range = np.zeros(self.value_param)
        self._summ_dtw_range()

        self.psi = None
        self._psi()

        self.d_theshold = self.summ_dtw_range / self.value_sign+3 * self.psi

    def tuple_confusion(self) -> tuple[int]:
        tp = 0
        tn = 0

        for i in self.test_true:
            if self.predict(i):
                tp += 1

        for i in self.test_fake:
            if not self.predict(i):
                tn += 1

        return tp, tn, len(self.test_true)-tp, len(self.test_fake)-tn

    def predict(self, data) -> bool:
        result = np.zeros(self.value_param)
        for p in range(self.value_param):
            result[p] = dtw(data[p], self.g_ref[p])

        return (result <= self.d_theshold).all()

    def compute_dtw_barycenter(self, params):
        reference = params[0]
        aligned_params = [reference]

        for p in params[1:]:
            path, _ = dtw_path(reference, p)
            aligned_p = np.zeros(len(reference))
            count = np.zeros(len(reference))

            for i, j in path:
                aligned_p[i] += p[j]
                count[i] += 1

            aligned_p /= np.where(count == 0, 1, count)
            aligned_params.append(aligned_p)

        return np.mean(aligned_params, axis=0)

    def _summ_dtw_range(self):
        for sign in self.train_true:
            for param in range(self.value_param):
                self.summ_dtw_range[param] += dtw(self.g_ref[param], sign[param])

    def _g_ref(self):
        for mp in self.mass_params:
            self.g_ref.append(self.compute_dtw_barycenter(mp))

    def _psi(self):
        psi_2 = np.zeros(self.value_param)
        d_th = self.summ_dtw_range / self.value_sign

        for sign in self.train_true:
            for param in range(self.value_param):
                psi_2[param] += (dtw(self.g_ref[param],sign[param].reshape(-1,1)) - d_th[param])**2 / self.value_sign
        self.psi = np.sqrt(psi_2)