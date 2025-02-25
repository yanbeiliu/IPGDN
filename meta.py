#!/usr/bin/env python3
#
# This is the script I use to tune the hyper-parameters automatically.
#
import subprocess

import hyperopt
import re
import sys

min_y = 0
min_c = None


class Output(object):
    # 控制台内容生成txt报告
    def __init__(self, check_filename="default.log"):
        self.terminal = sys.stdout
        self.log = open(check_filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):         # 即时更新
        pass


def trial(hyperpm):
    global min_y, min_c
    # Plz set nbsz manually. Maybe a larger value if you have a large memory.
    cmd = 'python main.py --datname Cora --nbsz 30'
    # cmd = 'CUDA_VISIBLE_DEVICES=0 ' + cmd
    for k in hyperpm:
        v = hyperpm[k]
        cmd += ' --' + k
        if int(v) == v:
            cmd += ' %d' % int(v)
        else:
            cmd += ' %g' % float('%.1e' % float(v))
    try:
        # val, tst = eval(subprocess.check_output(cmd, shell=True))
        btem = subprocess.check_output(cmd, shell=True)
        bstr = re.findall(r"\d+\.?\d*", btem.decode())
        val = float(bstr[0])
        tst = float(bstr[1])
    except subprocess.CalledProcessError:
        print('...')
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL}
    print('val=%5.2f%% tst=%5.2f%% @ %s' % (val * 100, tst * 100, cmd))
    score = -tst
    if score < min_y:
        min_y, min_c = score, cmd
    return {'loss': score, 'status': hyperopt.STATUS_OK}


space = {'lr': hyperopt.hp.loguniform('lr', -6, 0),  
         'reg': hyperopt.hp.loguniform('reg', -10, -2),  
         'nlayer': hyperopt.hp.quniform('nlayer', 3, 8, 1),  
         'ncaps': 4,
         'nhidden': 16,
         # 'ncaps': hyperopt.hp.quniform('ncaps', 3, 8, 1),    
         # 'nhidden': hyperopt.hp.quniform('nhidden', 10, 32, 2),  
         'dropout': hyperopt.hp.uniform('dropout', 0, 1),
         'routit': 7,
         #'lamda': 0}
         'lamda': hyperopt.hp.uniform('lamda', 1e-7, 1e-5)}  

hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=100)
print('>>>>>>>>>> tst=%5.2f%% @ %s' % (-min_y * 100, min_c))
Output('%s.txt' % 'Cora')
