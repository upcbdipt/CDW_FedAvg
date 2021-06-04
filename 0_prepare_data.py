# -*- coding: utf-8 -*-

# @Time  : 2021/6/4 上午11:22
# @Author : upcbdipt
# @Project : CDW_FedAvg
# @FileName: 0_prepare_data

from main.preprogressor import prepare_client_data, prepare_server_data

if __name__ == '__main__':
    clients_data_file_name = ['F1_Equal.npy', 'F2_Equal.npy', 'F3_Equal.npy', 'F4_Equal.npy']
    # prepare train data and test data of each client
    for client_data_file_name in clients_data_file_name:
        prepare_client_data(client_data_file_name)
    # prepare train data and test data of the server
    prepare_server_data(clients_data_file_name)
