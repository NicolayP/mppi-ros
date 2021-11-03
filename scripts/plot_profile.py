import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_profile(filename):
    pass


if __name__ == "__main__":
    plot_profile("foo")

file = "/home/pierre/workspace/uuv_ws/src/mppi-ros/log/profile.yaml"

with open(file, 'r') as stream:
    profile = yaml.safe_load(stream)

avg_step = profile['total']/profile['calls']
avg_rand = profile['rand']/profile['calls']
avg_update = profile['update']/profile['calls']

rollout_dict = profile['rollout']
avg_rollout = rollout_dict['total']/rollout_dict['calls']
avg_cost = rollout_dict['cost']/rollout_dict['calls']


rollout_avg_cost = rollout_dict['cost']/rollout_dict['calls']
rollout_avg_model = rollout_dict['model']['total']/rollout_dict['calls']
rollout_avg_total = rollout_dict['total']/rollout_dict['calls']


model_dict = rollout_dict['model']

avg_model_acc = model_dict['acc']['total']/rollout_dict['calls']
avg_model_b2i = model_dict['b2i_trans']/rollout_dict['calls']
avg_model_pose_dot = model_dict['pose_dot']/rollout_dict['calls']
avg_model_total = model_dict['total']/rollout_dict['calls']


acc_dict = model_dict['acc']

acc_avg_cori = acc_dict['cori']/rollout_dict['calls']
acc_avg_damp = acc_dict['damp']/rollout_dict['calls']
acc_avg_rest = acc_dict['rest']/rollout_dict['calls']
acc_avg_solv = acc_dict['solv']/rollout_dict['calls']
acc_avg_total = acc_dict['total']/rollout_dict['calls']



labels = ['total', 'rollout', 'model', 'acc']
rand = np.array([avg_rand, 0., 0., 0.])
update = np.array([avg_update, 0., 0., 0.])
rollout = np.array([avg_rollout, 0., 0., 0.])
div_tot = np.array([avg_step-avg_rand-avg_update-avg_rollout, 
           rollout_avg_total-rollout_avg_cost-rollout_avg_model,
           avg_model_total-avg_model_b2i-avg_model_pose_dot-avg_model_acc,
           acc_avg_total-acc_avg_cori-acc_avg_damp-acc_avg_rest-acc_avg_solv])

roll_cost = np.array([0., rollout_avg_cost, 0., 0.])
roll_model = np.array([0., rollout_avg_model, 0., 0.])

model_acc = np.array([0., 0., avg_model_acc, 0.])
model_b2i = np.array([0., 0., avg_model_b2i, 0.])
model_pose_dot = np.array([0., 0., avg_model_pose_dot, 0.])

acc_cori = np.array([0., 0., 0., acc_avg_cori])
acc_damp = np.array([0., 0., 0., acc_avg_damp])
acc_rest = np.array([0., 0., 0., acc_avg_rest])
acc_solv = np.array([0., 0., 0., acc_avg_solv])

print("*"*5 + " Total " + "*"*5)
print((div_tot)[0])
print((div_tot+rand)[0])
print((div_tot+rand+update)[0])
print((div_tot+rand+update+rollout)[0])

print("*"*5 + " Rollout " + "*"*5)
print((div_tot+rand+update+rollout+roll_cost)[1])
print((div_tot+rand+update+rollout+roll_cost+roll_model)[1])

print("*"*5 + " Model " + "*"*5)
print((div_tot+rand+update+rollout+roll_cost+roll_model+model_acc)[2])
print((div_tot+rand+update+rollout+roll_cost+roll_model+model_acc+model_b2i)[2])
print((div_tot+rand+update+rollout+roll_cost+roll_model+model_acc+model_b2i+model_pose_dot)[2])

print("*"*5 + " Acc " + "*"*5)
print((div_tot+rand+update+rollout+roll_cost+roll_model+model_acc+model_b2i+model_pose_dot+acc_cori)[3])
print((div_tot+rand+update+rollout+roll_cost+roll_model+model_acc+model_b2i+model_pose_dot+acc_cori+acc_damp)[3])
print((div_tot+rand+update+rollout+roll_cost+roll_model+model_acc+model_b2i+model_pose_dot+acc_cori+acc_damp+acc_rest)[3])
print((div_tot+rand+update+rollout+roll_cost+roll_model+model_acc+model_b2i+model_pose_dot+acc_cori+acc_damp+acc_rest+acc_solv)[3])

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots(figsize=(15,15))

ax.bar(labels, rollout, width, label='rollout')
ax.bar(labels, rand, width, bottom=rollout, label='rand')
ax.bar(labels, update, width, bottom=rand+rollout, label='update')

ax.bar(labels, roll_model, width, label='model')
ax.bar(labels, roll_cost, width, bottom=roll_model, label='cost')

ax.bar(labels, model_acc, width, label='acc')
ax.bar(labels, model_pose_dot, width, bottom=model_acc, label='p_dot')
ax.bar(labels, model_b2i, width, bottom=model_acc+model_pose_dot, label='b2i')

ax.bar(labels, acc_cori, width, label='cori')
ax.bar(labels, acc_damp, width, bottom=acc_cori, label='damp')
ax.bar(labels, acc_rest, width, bottom=acc_damp+acc_cori, label='rest')
ax.bar(labels, acc_solv, width, bottom=acc_rest+acc_damp+acc_cori, label='solv')

ax.bar(labels, div_tot, width, bottom=acc_solv+acc_rest+acc_damp+acc_cori+model_acc+model_b2i+model_pose_dot+roll_model+roll_cost+rollout+update+rand, label='diverse', color='black')

ax.set_ylabel('Time (s)')
ax.set_title('MPPI profiling')
ax.legend(bbox_to_anchor=(1.1, 1.05))


plt.show()
