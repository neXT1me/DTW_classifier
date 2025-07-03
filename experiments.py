import numpy as np
import pandas as pd

from DTW import DTW_classification

from tqdm import tqdm
import random

def get_data(data):
    df_sorted = data.sort_values(by=['id_sign', 'pktID'])
    grouped = df_sorted.groupby('id_sign')
    parameters = df_sorted.columns.difference(['id_sign', 'id_sign.1', 'id_user', 'pktID']).tolist()

    data_sign = [
        np.array([
            group[param].tolist()

            for param in parameters
        ])
        for _, group in grouped
    ]
    return data_sign

def experiment(data_train, data_fake_test, data_true_test) -> list:
    '''
    Функция для формирования списка кортежей из элементов матрицы ошибок при разном количестве входных подписей
    для обучении модели.

    :param data_train: обучающий набор данных основанный на подлинных подписях.
    :param data_fake_test: тестовый набор данных основанный на поддельных подписях.
    :param data_true_test: тестовый набор данных основанный на подлинных подписях.
    :return: список с корежами элеметов матрицы ошибок (tp, tn, fp, fn).
    '''
    n = len(data_train)
    accuracy_param = []

    for i in tqdm(range(1, n + 1)):
        model = DTW_classification(data_train[:i], data_fake_test, data_true_test)
        accuracy_param.append(model.tuple_confusion())
    return accuracy_param

if __name__ == '__main__':
    file = 'sign_data.csv'
    df = pd.read_csv(file)


    # col = ['id_user', 'id_sign', 'pktID', 'X', 'Y', 'Z',
    #                       'pkNormalPressure',
    #                       'pkOrientationOrAltitude',
    #                       'pkOrientationOrAzimuth']
    # param = ['id_user', 'id_sign', 'pktID', 'X', 'Y', 'Z',
    #                       'pkNormalPressure',
    #                       'pkOrientationOrAltitude',
    #                       'pkOrientationOrAzimuth']

    list_id = [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 22]

    # Задача бинарной классификации, условия:
    # 1) Список используемых id [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 22]
    # 2) Для каждого id используется 20 экземпляров данных
    # 3) количество тестовой выборки 50

    id = list_id[0]
    df_sign_true = df[df.id_user == id]
    df_sign_fake = df[df.id_user != id]


    data_sign_true = get_data(df_sign_true)
    data_sign_fake = get_data(df_sign_fake)
    # ---------------------- Тестирование при value_sign_true = 20 ----------------------------------
    list_acc = []
    n = 10

    for i in tqdm(range(n)):
        dt = random.sample(data_sign_true, k=20)
        dft = random.sample(data_sign_fake, k=50)
        dtt = random.sample(data_sign_true, k=50)
        result = experiment(data_train=dt,
                        data_fake_test=dft,
                        data_true_test=dtt)
        list_acc.append(result)

    with open('result_20_50_50_tft.txt', 'w') as f:
        f.write(str(list_acc))