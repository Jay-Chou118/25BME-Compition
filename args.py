"""
作者：yueyue
日期：2023年10月30日
"""
import os
import argparse
parser = argparse.ArgumentParser()
home_dir = os.getcwd()
parser.add_argument('--save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device_cuda', type=int, default=2, help='GPU')
parser.add_argument('--base_model', default='MACNN', type=str,
                    help='The Base Feature extractor to be used')
args = parser.parse_args()
