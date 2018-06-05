#--- Импорт пакетов
import sys  # sys нужен для передачи argv в QApplication
import os  # Отсюда нам понадобятся методы для отображения содержимого директорий
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams
rcParams['font.family'] = 'Georgia'
rcParams['font.size'] = 8
import matplotlib.pyplot as plt
import warnings
from itertools import product
import pandas_datareader.data as web
from PyQt5 import QtWidgets, QtGui, QtCore
import datetime
from arch import arch_model
import datetime
from dateutil.relativedelta import relativedelta
import mainwindow # Это наш конвертированный файл дизайна
#-----

class PandasModel(QtCore.QAbstractTableModel): 
    def __init__(self, df = pd.DataFrame(), parent=None): 
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.ix[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit() 

class MainApp(QtWidgets.QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле .py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        # при нажатии кнопки
        self.pushButton.clicked.connect(self.download) # обработка события нажатия на кнопку
        # отрисовка полей для графиков
        self.figure = Figure(figsize=(5,4))
        self.canvas=FigureCanvas(self.figure)
        self.gridLayout_4.addWidget(self.canvas)
        self.figure1 = Figure()
        self.canvas1=FigureCanvas(self.figure1)
        self.gridLayout_6.addWidget(self.canvas1)
        #-----
    def verify(self):
        start = self.dateEdit.date()
        start = start.toPyDate()
        end = self.dateEdit_2.date()
        end=end.toPyDate()
        symbol = self.lineEdit.text() 

    def download(self): 
        start = self.dateEdit.date()
        start = start.toPyDate()
        end = self.dateEdit_2.date()
        end=end.toPyDate()
        symbol = self.lineEdit.text()
        data = web.DataReader(symbol,'morningstar', start=start, end=end)['Close']
        dt = pd.DataFrame(data)
        logrets = np.log(data / data.shift(1)).dropna()
        self.tsplot(data[symbol], symbol)
        if self.checkBox.isChecked():
            data= np.log(data / data.shift(1)).dropna()
        if self.sdiffcheckBox.isChecked():
            difflog=self.sdiffspinBox.text()
            difflog=int(difflog)
            data=(data-data.shift(difflog)).dropna()
        if self.dispcheckBox.isChecked():
            data[symbol], lmbda = stats.boxcox(data[symbol])
        self.tsplot(data[symbol],symbol)
        self.label_dikiful.setText("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(data[symbol])[1])
        if self.radioButton_arima.isChecked():
            res_tup=self.getbestarima(data,symbol)
        if self.radioButton_ar.isChecked():
            self.getbestar(data,symbol)
        if self.radioButton_arch.isChecked():
            self.getbestarima(data,symbol)

    def getbestar(self,data,symbol):
        max_lag = 30
        mdl = smt.AR(data[symbol]).fit(maxlag=max_lag, ic='aic', trend='nc')
        best_order = smt.AR(data[symbol]).select_order(
        maxlag=max_lag, ic='aic', trend='nc')
        self.label_dikiful_2.setText('best estimated lag order = {}'.format(best_order))
        max_lag = best_order
        Y = data[symbol]
        best_mdl = smt.ARMA(Y, order=(0, 3)).fit(
        maxlag=max_lag, method='mle', trend='nc')
        if self.checkBox_forecast.isChecked():
            self.forecast(data,symbol,best_mdl,int(self.sdiffspinBox_2.text()))
        elif not self.checkBox_forecast.isChecked():
            self.tsplot(best_mdl.resid, symbol)

    def getbestarima(self,data,symbol):
        best_aic = np.inf
        best_order = None
        best_mdl = None

        pq_rng = range(5)  # [0,1,2,3,4]
        d_rng = range(2) # [0,1]
        for i in pq_rng:
            for d in d_rng:
                for j in pq_rng:
                    try:
                        tmp_mdl = smt.ARIMA(data.get(symbol), order=(i, d, j)).fit(method='mle', trend='nc')
                        tmp_aic = tmp_mdl.aic
                        if tmp_aic < best_aic:
                            best_aic = tmp_aic
                            best_order = (i, d, j)
                            best_mdl = tmp_mdl
                    except:continue

        self.label_dikiful_2.setText('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
        if self.checkBox_forecast.isChecked() & self.radioButton_arima.isChecked():
            self.forecast(data, symbol, best_mdl, int(self.sdiffspinBox_2.text()))
        elif not self.checkBox_forecast.isChecked() & self.radioButton_arima.isChecked():
            self.tsplot(best_mdl.resid, symbol)

        return best_aic, best_order, best_mdl


    def getbestarch(self,data,symbol):
        res_tup=self.getbestarima(data,symbol)
        (best_aic, best_order, best_mdl)=res_tup
        p_ = best_order[0]
        o_ = best_order[1]
        q_ = best_order[2]
        # Using student T distribution usually provides better fit
        am = arch_model(data[symbol], p=p_, o=o_, q=q_, dist='StudentsT')
        res = am.fit(update_freq=5, disp='off')
        self.tsplot(res.resid, symbol)
        if self.checkBox_forecast.isChecked():
            self.forecastarch(data,symbol,res,best_order,int(self.sdiffspinBox_2.text()))

    def forecastarch(self, data,symbol,res, best_order,h):
        y = data
        mu = res.params['mu']
        omega = res.params['omega']
        p = best_order[0]
        q = best_order[2]
        alpha = res.params[2:2 + p]
        beta = res.params[2 + p:2 + p + q]
        start_index = len(data)
        horizont = h
        y_del = y[0: start_index]
        if alpha is not None:
            alpha = alpha[::-1]
        else:
            alpha = 0
        if beta is not None:
            beta = beta[::-1]
        else:
            beta = 0
        for h in range(1, horizont):
            y_arr_p = (y_del[-p:] - mu).values
            a_e = np.sum(alpha * y_arr_p ** 2)
            b_o = np.sum(beta * np.power(y_del[-q:] - mu, 2))
        date_list = [datetime.datetime.strptime(str(end), "%Y-%m-%d") + relativedelta(days=x) for x in range(0, h + 1)]
        forecast= pd.Series([mu + np.sqrt(omega + a_e)], index=date_list[1:])
        self.plotforecast(self, y, forecast, symbol, h)

    def forecast(self, data,symbol,best_mdl,h=10):
        end = self.dateEdit_2.date()
        end = end.toPyDate()
        data2 = data.get(symbol)
        #end=str(end)[6:10]+'-'+str(end)[0:2]+'-'+str(end)[3:5]
        date_list = [datetime.datetime.strptime(str(end), "%Y-%m-%d") + relativedelta(days=x) for x in
                     range(0, h+1)]
        forecast=best_mdl.predict(start=len(data), end=len(data)-1+h)
        forecast.index=date_list[1:]
        #data2['forecast'] = best_mdl.forecast(steps=h-1)
        self.plotforecast(data,forecast,symbol,h)
        df=pd.DataFrame([forecast.index,forecast]).transpose()
        df.columns=['Date',symbol.upper()]
        model = PandasModel(df)
        self.tableView.setModel(model)

    def plotforecast(self,data,forecast,symbol,h=10):        
        #ax2=self.figure1.add_subplot(2,1,2)
        self.ax1=self.figure1.add_subplot(1,1,1)
        self.ax1.clear()
        #ax2.clear()
        data[symbol][-10-h:].plot(ax=self.ax1)
        forecast[-h:].plot(ax=self.ax1,color='r')
        self.canvas1.draw()

    def tsplot(self,y,symbol):
        self.ts_ax = self.figure.add_subplot(3,1,1)
        self.acf_ax = self.figure.add_subplot(3, 2, 3)
        self.pacf_ax = self.figure.add_subplot(3, 2, 4)
        self.qq_plot_ax = self.figure.add_subplot(3, 2, 5)
        self.pp_plot_ax = self.figure.add_subplot(3, 2, 6)
        self.ts_ax.clear()
        self.acf_ax.clear()
        self.pacf_ax.clear()
        self.qq_plot_ax.clear()
        self.pp_plot_ax.clear()
        self.ts_ax.plot(y)
        self.ts_ax.set_title('Time Series Analysis Plots for ' + symbol.upper())
        smt.graphics.plot_acf(y, lags=30, ax=self.acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=30, ax=self.pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=self.qq_plot_ax)
        self.qq_plot_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=self.pp_plot_ax)
        self.canvas.draw()


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = MainApp()  # Создаём объект класса MainApp
    window.showMaximized()  # Показываем окно
    app.exec_()  # и запускаем приложение

if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()


   