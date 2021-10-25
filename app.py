from PyQt5 import QtCore, QtGui, QtWidgets, QtPrintSupport, uic
import os, pandas as pd, sys
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from keras.layers.normalization import BatchNormalization
from keras.layers import LayerNormalization
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping


# ui_file, _ = loadUiType(path.join(path.dirname(__file__), "App2.ui"))


class Main(QtWidgets.QWidget):
    def __init__(self):
        super(Main, self).__init__()
        uic.loadUi(os.path.join(os.getcwd(), "App2.ui"), self)

        self.textBrowser.setFixedHeight(40)
        self.push.setStyleSheet(stylesheet(self))
        self.push1.setStyleSheet(stylesheet(self))
        self.push2.setStyleSheet(stylesheet(self))
        self.push3.setStyleSheet(stylesheet(self))
        self.push4.setStyleSheet(stylesheet(self))
        self.push5.setStyleSheet(stylesheet(self))

        self.push2.clicked.connect(self.Reg)
        self.push3.clicked.connect(self.Rms)
        self.push4.clicked.connect(self.showCSV)
        self.push5.clicked.connect(self.R_square)
        self.label.setPixmap(QtGui.QPixmap('umpGEP.png')) #todo:  debri 3la had l picture o diriha f nefss directory m3a app.py
        self.push1.clicked.connect(self.ANN)
        self.length = 0

        self.push.clicked.connect(self.loadCsv)
        self.push1.setEnabled(False)


        self.label.setScaledContents(True)
        self.label.setFixedHeight(100)
        self.label.setFixedWidth(500)
        self.setWindowTitle('Fouling with ANN')
        self.setWindowIcon(QtGui.QIcon('gep.png')) #todo: o hta hadi
        self.dataset = pd.DataFrame()

    def R_square(self):
        # plot training curve for R^2 (beware of scale, starts very low negative)
        plt.plot(self.result.history['val_r_square'])
        plt.plot(self.result.history['r_square'])
        plt.title('model R^2')
        plt.ylabel('R^2')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def showCSV(self):
        import os
        os.startfile(self.newFile)
        QtWidgets.QMessageBox.about(self, "Message", self.newFile + " Created !!")

    def Rms(self):
        # plot training curve for rmse
        plt.plot(self.result.history['rmse'])
        plt.plot(self.result.history['val_rmse'])
        plt.title('RMSE')
        plt.ylabel('RMSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def Reg(self):
        # print the linear regression and display datapoints

        from sklearn.linear_model import LinearRegression

        regressor = LinearRegression()
        regressor.fit(self.y_test.reshape(-1, 1), self.y_pred)
        y_fit = regressor.predict(self.y_pred)

        reg_intercept = round(regressor.intercept_[0], 4)
        reg_coef = round(regressor.coef_.flatten()[0], 4)
        reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)
        plt.scatter(self.y_test, self.y_pred, color='blue', label='data')
        plt.plot(self.y_pred, y_fit, color='red', linewidth=2, label='Linear regression\n' + reg_label)
        plt.title('Linear Regression')
        plt.legend()
        plt.xlabel('observed')
        plt.ylabel('predicted')
        plt.show()

    def ANN(self):
        #self.dataset = pd.read_csv(self.fileName, sep=";", decimal=",")

        x = self.dataset.iloc[:, :7].values
        y = self.dataset.iloc[:, 7].values
        self.X = x

        # Normalization

        sc = MinMaxScaler()
        x = sc.fit_transform(x)

        y = y.reshape(-1, 1)
        y = sc.fit_transform(y)

        # Splitting the dataset into the Training set and Test set
        x_train, x_test, y_train, self.y_test = train_test_split(x, y, test_size=0.10, random_state=4)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.05, random_state=4)

        # built Keras sequential model
        model = Sequential()
        # add batch normalization
        model.add(LayerNormalization())
        # Adding the input layer:
        model.add(Dense(7, input_dim=x_train.shape[1], activation='relu'))
        # the first hidden layer:
        model.add(Dense(12, activation='relu'))
        # Adding the output layer
        model.add(Dense(1, activation='sigmoid'))

        # Compiling the ANN
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', rmse, r_square])

        # enable early stopping based on mean_squared_error
        earlystopping = EarlyStopping(monitor="mse", patience=40, verbose=1, mode='auto')

        # Fit model
        self.result = model.fit(x_train, y_train, batch_size=100, epochs=1000, validation_data=(x_test, self.y_test),
                                callbacks=[earlystopping])

        # result = model.fit(x_train, y_train, batch_size=100, epochs=1000, validation_data=(x_test, y_test),
        #                   callbacks=[earlystopping])

        # get predictions
        self.y_pred = model.predict(x_test)
        self.y_x_pred = model.predict(x)
        self.Y = sc.inverse_transform(self.y_x_pred)

        # -----------------------------------------------------------------------------
        # print statistical figures of merit
        # -----------------------------------------------------------------------------
        import sklearn.metrics, math
        self.textBrowser_2.setText(
            f"Mean absolute error (MAE):     {sklearn.metrics.mean_absolute_error(self.y_test, self.y_pred)}" )
        self.textBrowser_2.append(
            f"Mean squared error (MSE):     {sklearn.metrics.mean_squared_error(self.y_test, self.y_pred)}")
        self.textBrowser_2.append(f"Root mean squared error (RMSE): {math.sqrt( sklearn.metrics.mean_squared_error(self.y_test, self.y_pred))}")
        self.textBrowser_2.append(
            f"R square (R^2):                 {sklearn.metrics.r2_score(self.y_test, self.y_pred)}")

        self.dataset['Thwo_Predict'] = self.Y
        self.newFile = "newCsvFile.xlsx"
        self.dataset.to_excel(self.newFile, index=False)

    def loadCsv(self):
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV",
                                                                 ".",
                                                                 "CSV (*.csv *.tsv )")
        if self.fileName and os.path.isfile(self.fileName):
        
            #self.push1.setEnabled(True)
            self.textBrowser.setText(os.path.basename(self.fileName))
            self.dataset.append(pd.read_csv(self.fileName, sep=";", decimal=","), ignore_index=True)
            # self.length = self.length + len(dataset.columns)
            if len(self.dataset.columns) == 8:
                """
                if self.length < 8:
                    print("length = ",self.length ," < 8")
                    self.push1.setEnabled(False)
                    self.push.setText("Add file")
                else:
                    print("length = ",self.length, " > 8")
                    print("You have exceeded the maximum size, 8 columns! ")
                    self.push1.setEnabled(False)
		        """
                self.push1.setEnabled(True)
                self.push.setText("Open CSV file")
            elif len(self.dataset.columns) > 8:
                self.push.setText(f"ERROR : {len(self.dataset.columns)} columns found, 8 expected")

            else:

                self.push.setText("Add file")





# -----------------------------------------------------------------------------
# Define custom loss functions for regression in Keras
# -----------------------------------------------------------------------------

# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (1 - SS_res / (SS_tot + K.epsilon()))


def stylesheet(self):
    return """


QPushButton::hover
{
border: 2px inset goldenrod;
font-weight: bold;
color: #e8e8e8;
background-color: green;
} 
"""


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationDisplayName("Fouling with ANN")
    main = Main()
    main.show()
    sys.exit(app.exec_())
