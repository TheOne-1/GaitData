import xlrd
import matplotlib.pyplot as plt
from Drawer import Drawer


date = '1030'
start_range, end_range = range(-24, 13, 4), range(2, 35, 4)

result_start, result_end = [], []
for clip_start in start_range:
    file_path = 'result_conclusion/' + date + '/trial_summary/start_' + str(clip_start) + '_end_' + str(30) + '.xlsx'
    sheet = xlrd.open_workbook(file_path).sheet_by_index(0)
    result_start.append(sheet.cell_value(61, 10))
for clip_end in end_range:
    file_path = 'result_conclusion/' + date + '/trial_summary/start_' + str(-12) + '_end_' + str(clip_end) + '.xlsx'
    sheet = xlrd.open_workbook(file_path).sheet_by_index(0)
    result_end.append(sheet.cell_value(61, 10))

Drawer.show_clip_sensitivity(start_range, result_start)
Drawer.show_clip_sensitivity(end_range, result_end, color='r', name='End', used_loc=7)
plt.show()
