from const import PROCESSED_DATA_PATH, RAW_DATA_PATH, FILE_NAMES
from ParameterProcessor import ParamProcessor

subject_folder = '190414WangDianxin'
readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme.xlsx'
my_initializer = ParamProcessor(PROCESSED_DATA_PATH, subject_folder, readme_xls)


# with open(data_path + 'subject_info.csv') as subject_info_file:
#     reader = csv.reader(subject_info_file)
#     for row in reader:  # there is only one row in the csv file
#         subject_info = row
#
# # dynamic_var = locals()  # dynamically name the Object and csv file
# for i_trial in range(1, 8):
#     # gait_data_len = gait_data.shape[0]
#     file_path = data_path + TRIAL_NAMES[i_trial] + '.csv'
#     parameter_processor = ParamProcessor(file_path, float(subject_info[1]), float(subject_info[2]), sub_name)
#
#     heel_strikes = parameter_processor.get_heel_strike_event()
#     toe_offs = parameter_processor.get_toe_off_event()
#     FPA_all = parameter_processor.get_FPA_all()    # FPA of all the samples
#     max_LR_all = parameter_processor.get_loading_rate(sub_name, i_trial)
#     trunk_swag = parameter_processor.get_trunk_swag()
#     strike_index_all = parameter_processor.get_strike_index_all()
#     strike_angle_all = parameter_processor.get_foot_strike_angle()
#     strike_off_from_IMU = parameter_processor.get_strike_off_from_imu(i_trial)
#
#     # # check data
#     # parameter_processor.check_strikes_off(heel_strikes, toe_offs, check_len=-1)
#     # parameter_processor.check_trunk_swag(trunk_swag)
#     # parameter_processor.check_FPA_all(FPA)
#     # plt.show()
#
#     # transfer matrix to DataFrame
#     raw_data = np.column_stack([heel_strikes, toe_offs, FPA_all, max_LR_all, trunk_swag, strike_index_all,
#                                 strike_angle_all, strike_off_from_IMU])
#     data = pd.DataFrame(raw_data, columns=[
#         'l_heel_strikes', 'r_heel_strikes', 'l_toe_offs', 'r_toe_offs', 'l_FPA_all', 'r_FPA_all',
#         'l_max_LR', 'r_max_LR', 'trunk_swag', 'l_strike_index_all', 'r_strike_index_all',
#         'l_strike_angle', 'r_strike_angle', 'l_strike_from_IMU', 'l_off_from_IMU'])
#
#     object_name = 'params_of_' + TRIAL_NAMES[i_trial]  # the name of the GaitData object and file
#     current_class = GaitParameter(object_name)  # initial the object
#     save_path = data_path + '\\'
#     current_class.set_data(data)  # pass the data to the matrix
#     # save as csv
#     current_class.save_as_csv(save_path)  # save as csv
#     current_class.clear_old_csv(save_path, object_name)  # delete the former file




