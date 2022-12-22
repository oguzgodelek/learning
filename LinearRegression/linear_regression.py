import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=10e-9, n_iterations=100, batch_size=8, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.verbose = verbose
        self.data = None
        self.W = None
        self.y = None
        self.error_history = []

    def fit(self, data, y):
        self.data = data
        self.data['bias'] = 1.0
        self.y = np.matrix(y)
        self.W = np.random.randint(0, 10, (len(self.data.columns), 1)) / 10000
        print(self.W)
        y_chunks = list(map(lambda x: np.matrix(x), self.split_dataframe(y)))
        x_chunks = self.split_dataframe(self.data)
        for epoch in range(self.n_iterations):
            if epoch % 10 == 0:
                self.learning_rate /= 2
            for batch_n, batch in enumerate(x_chunks):
                error = self.calculate_mse_error()
                self.error_history.append(error)
                if self.verbose:
                    print("Epoch:", epoch+1, ", Batch:", batch_n+1, "MSE Error:", error)
                predictions = self.predict(batch)
                self.W -= self.learning_rate * np.dot(batch.T, predictions - y_chunks[batch_n].T) \
                          / batch.shape[0]
            # error = self.calculate_mse_error()
            # self.error_history.append(error)
            if len(self.error_history) > 2 and abs(self.error_history[-1] - self.error_history[-2]) < 1e-2:
                break

    def predict(self, x):
        if self.W is None:
            return None
        return np.dot(x, self.W)

    def calculate_mse_error(self):
        if self.data is None:
            return None
        predictions = self.predict(self.data)
        return np.sum(np.square(predictions - self.y.T)) * 0.5 / len(self.data)

    def split_dataframe(self, dataframe):
        splitted_df = [dataframe.iloc[i:i+self.batch_size+1] for i in range(int(dataframe.shape[0] / self.batch_size))]
        splitted_df.append(dataframe[int(dataframe.shape[0] / self.batch_size) * self.batch_size:])
        return splitted_df


def main():
    df = pd.read_csv('Real estate.csv')
    train_y = df['Y house price of unit area']
    train_x = df.drop(['Y house price of unit area', 'No'], axis=1)
    linear_regressor = LinearRegression()
    linear_regressor.fit(train_x, train_y)
    plt.plot(linear_regressor.error_history)
    plt.xlabel('Batch number')
    plt.ylabel('MSE')
    plt.title('MSE Error Loss During Training')
    plt.show()


if __name__ == '__main__':
    main()
