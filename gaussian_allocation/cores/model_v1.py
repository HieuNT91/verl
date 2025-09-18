import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

def kernel_rbf(d, length_scale):
    return np.exp(-0.5 * (d / length_scale) ** 2)

def kernel_rbf_median(d):
    length_scale = np.median(d[np.triu_indices_from(d,k=1)].nonzero())
    return np.exp(-0.5 * (d / length_scale) ** 2)

PRIOR_VALUE = -5
class SequentialGPR:
    def __init__(self, 
                 distance_matrix, 
                 length_scale=1.0,
                 reuse_covariance=False,
                 reuse_mean=False,
                 return_std=False
                 ):
        self.distance_matrix = distance_matrix
        self.length_scale = length_scale
        self.covariance_matrix = kernel_rbf_median(distance_matrix)
        self.mean = np.zeros(self.covariance_matrix.shape[0])  + PRIOR_VALUE
        self.reuse_covariance = reuse_covariance
        self.reuse_mean = reuse_mean
        self.return_std = return_std
        
    def _logit(self, p, eps=1e-6):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))
    
    def _sigmoid(self, f):
        return 1.0 / (1.0 + np.exp(-f))
    
    
    def fit(self, train_indices, observations):
        # self.covariance_matrix = kernel_rbf(self.distance_matrix, length_scale=self.length_scale)
        if not self.reuse_mean:
            self.mean = np.zeros(self.covariance_matrix.shape[0])  + PRIOR_VALUE
        # g_t = self._logit(observations)
        g_t = self._logit(np.clip(observations , a_max=1-1e-6, a_min=1e-6))
        # mean_in = self.mean[train_indices]
        K_in_in = self.covariance_matrix[np.ix_(train_indices, train_indices)]

        L = cholesky(K_in_in + 1e-6 * np.eye(K_in_in.shape[0]), lower=True, check_finite=False)
        alpha = cho_solve((L, True), g_t - self.mean[train_indices], check_finite=False)

        # y_centered = g_t - mean_in
        # K_in_in_inv = np.linalg.inv(K_in_in + 1e-6 * np.eye(len(train_indices)))
        # alpha_ = K_in_in_inv @ y_centered
        # assert np.allclose(alpha, alpha_, atol=1e-4), "K_in_in @ alpha not close to y_centered"
        
        all_indices = list(range(self.covariance_matrix.shape[0]))
        # remaining_indices = [i for i in range(self.covariance_matrix.shape[0]) if i not in train_indices]
        K_in_new = self.covariance_matrix[np.ix_(train_indices, all_indices)]
        K_new_in = K_in_new.T
        
        # ALGO: update posterior 
        self.mean[all_indices] = self.mean[all_indices] + K_new_in @ alpha
        self.mean[train_indices] = g_t
        if self.return_std or self.reuse_covariance:
            # Y = solve_triangular(L, K_in_new, lower=True, check_finite=False)
            # V = solve_triangular(L.T, Y, lower=False, check_finite=False)
            # assert np.allclose(V, K_in_in_inv @ K_in_new, atol=1e-4), "Covariance update mismatch"
            K_in_in_inv = np.linalg.inv(K_in_in + 1e-6 * np.eye(len(train_indices)))
            V = K_in_in_inv @ K_in_new
            self.std = np.sqrt(np.diag(self.covariance_matrix[np.ix_(all_indices, all_indices)] - K_new_in @ V))
        
        if self.reuse_covariance:
            self.covariance_matrix[np.ix_(all_indices, all_indices)] -= K_new_in @ V


    def predict(self, indices):
        mean_pred = self._sigmoid(self.mean[indices])
        if self.return_std:
            return mean_pred, self.std[indices]
        else:
            return mean_pred, None


def acc_to_var(acc):
    return acc * (1 - acc)
    

if "__main__" == __name__:
    # Example usage
    # distance_matrix = np.load("/home/hieunt/verl/data/embedding_data/embeddings_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_1000.npy")
    # gpr = SequentialGPR(distance_matrix, length_scale=1.0)
    # train_indices = [1,2, 3, 4, 5, 6,]
    # observations = np.array([ 0.5, 0.5, 0.6, 0.5, 0.05, 0.5])
    # gpr.fit(train_indices, observations)
    # mean_pred, cov_pred = gpr.predict(train_indices)
    # print("[Train] Predicted means:", mean_pred)
    # print("[Train] Predicted covariances:", cov_pred)
    
    # test_indices = [13, 14, 15, 16, 17, 18, 19, 20]
    # mean_pred, cov_pred = gpr.predict(test_indices)
    # print("[Test] Predicted means:", mean_pred)
    # print("[Test] Predicted covariances:", cov_pred)
    
    # Test with time data 
    from time_data_simulator import TimeDataSimulator, TimeDataSimulatorConfig
    cfg = TimeDataSimulatorConfig(
        embedding_path="/home/hieunt/verl/data/embedding_data/embeddings_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_17398.npy",
        pairwise_path="/home/hieunt/verl/data/embedding_data/pairwise_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_17398_matrix.npy",
        indices_path="/home/hieunt/verl/data/embedding_data/indices_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_17398.json",
        regression_json_path="/home/hieunt/verl/data/regression_data/allo_grpo_4e/per_question_statistics_latest.json",
        batch_size=256,
    )
    sim = TimeDataSimulator(cfg)
    gpr = SequentialGPR(sim.pairwise_matrix, 
                        length_scale=0.6,
                        return_std=False,
                        reuse_covariance=False,
                        reuse_mean=False,
                        )
    
    
    
    
    
    
    for step in range(1, 120, 1):
        out = sim.get_train_test_features(
            step=step,
            window_size=1,
            target_key="mean_acc_per_epoch",
        )
        X_train, P_train, y_train = out["train"]["X"], out["train"]["P"], out["train"]["y"]
        X_test, P_test, y_test = out["test"]["X"], out["test"]["P"], out["test"]["y"]
        qids_train = out["train"]["qids"]
        qids_test = out["test"]["qids"]

        indices_train = out["train"]["indices"]
        indices_test = out["test"]["indices"]
        # print(f"Step {step}: Train size {X_train.shape}, Test size {X_test.shape}")
        # print(f"Train qids: {qids_train}")
        # print(f"Test qids: {qids_test}")    
        gpr.fit(indices_train, y_train)
        mean_pred_train, cov_pred_train = gpr.predict(indices_train)
        mean_pred_test, cov_pred_test = gpr.predict(indices_test)
        # print(f"[Train] Predicted means: {mean_pred_train}, True means: {y_train}")
        # print(f"[Test] Predicted means: {mean_pred_test}, True means: {y_test}")
        
        # measure mse and r2
        # mean_pred_test = np.array([np.mean(y_test)]*len(y_test))
        # mean_pred_test = np.array([0.5]*len(y_test))
        from sklearn.metrics import mean_squared_error, r2_score
        # mse_train = mean_squared_error(y_train, mean_pred_train)
        # r2_train = r2_score(y_train, mean_pred_train)
        # y_test = acc_to_var(y_test)
        # mean_pred_test = acc_to_var(mean_pred_test)
        mse_test = mean_squared_error(y_test, mean_pred_test)
        r2_test = r2_score(y_test, mean_pred_test)
        almost_close_count = np.sum(
            (abs(y_test - mean_pred_test) <= 0.2)
        )

        # optional: convert to proportion if desired
        almost_close_ratio = almost_close_count / len(y_test)
        # print(f"Step {step} - Train MSE: {mse_train}, R2: {r2_train}")

        print(f"Step {step} - Test MSE: {mse_test:.3f}, R2: {r2_test:.3f}, Almost Close Ratio: {almost_close_ratio:.3f}, Min Pred: {mean_pred_test.min():.4f}, Max Pred: {mean_pred_test.max():.4f}, std Pred: {np.std(mean_pred_test):.4f}")
        # Print a few example predictions for test set
        # print("Example test predictions:")
        for i in range(min(5, len(mean_pred_test))):
            print(f"  y_true={y_test[i]:.4f}, y_pred={mean_pred_test[i]:.4f}")
        
        # linear regression 
        # from sklearn.linear_model import LinearRegression
        # lr = LinearRegression()
        # lr.fit(X_train, y_train)
        # y_pred_train_lr = lr.predict(X_train)
        # y_pred_test_lr = lr.predict(X_test)
        # mse_train_lr = mean_squared_error(y_train, y_pred_train_lr)
        # r2_train_lr = r2_score(y_train, y_pred_train_lr)
        # mse_test_lr = mean_squared_error(y_test, y_pred_test_lr)
        # r2_test_lr = r2_score(y_test, y_pred_test_lr)
        # almost_close_count_lr = np.sum(
        #     (abs(y_test - y_pred_test_lr) <= 0.1)
        # )
        # almost_close_ratio_lr = almost_close_count_lr / len(y_test)
        # # print(f"[Linear Regression] Step {step} - Train MSE: {mse_train_lr}, R2: {r2_train_lr}")
        # print(f"[Linear Regression] Step {step} - Test MSE: {mse_test_lr}, R2: {r2_test_lr}, Almost Close Ratio: {almost_close_ratio_lr:.4f}")
