import os

import numpy as np


def main():

    gpu_numbers = [0, 1, 2, 3, 4, 5, 6, 7]

    flags_sequence_bilstm = [True]
    word_in_fixation_orders = [True]
    use_reduced_pos_sequences = [True]
    use_content_word_sequences = [True]
    use_numerics = [True]
    use_fixation_sequences = [True]

    param_strings = []

    for flag_sequence_bilstm in flags_sequence_bilstm:
        for word_in_fixation_order in word_in_fixation_orders:
            for use_reduced_pos_sequence in use_reduced_pos_sequences:
                for use_content_word_sequence in use_content_word_sequences:
                    for use_numeric in use_numerics:
                        for use_fixation_sequence in use_fixation_sequences:
                            param_strings.append(
                                '-flag_sequence_bilstm ' + str(flag_sequence_bilstm) +
                                ' -word_in_fixation_order ' + str(word_in_fixation_order) +
                                ' -use_reduced_pos_sequence ' + str(use_reduced_pos_sequence) +
                                ' -use_content_word_sequence ' + str(use_content_word_sequence) +
                                ' -use_numeric ' + str(use_numeric) +
                                ' -use_fixation_sequence ' +
                                str(use_fixation_sequence),
                            )

    save_dir = 'nn/results/'
    os.makedirs(save_dir, exist_ok=True)
    script_path = 'nn/scripts/'
    os.makedirs(script_path, exist_ok=True)

    param_strings = np.random.permutation(param_strings)

    num_per_gpu = int(np.ceil(len(param_strings) / len(gpu_numbers)))

    script_texts = []
    counter = 0
    for gpu in gpu_numbers:
        gpu_commands = ''
        for i in range(num_per_gpu):
            if counter < len(param_strings):
                cur_command = 'python nn/model.py -GPU ' + str(gpu) + ' ' +\
                    param_strings[counter] + ' -save_dir ' +\
                    save_dir
                gpu_commands += cur_command + '\n'
                counter += 1
        script_texts.append(gpu_commands)

    # write scripts
    complete_run_text = ''
    for i in range(len(script_texts)):
        cur_text = script_texts[i]
        cur_save_path = script_path + 'gpu_' + str(gpu_numbers[i]) + '.sh'
        complete_run_text += cur_save_path + '&\n'
        fobj = open(cur_save_path, 'w')
        fobj.write(cur_text)
        fobj.close()
        # execute bit
        os.system('chmod +x ' + cur_save_path)
    complete_run_text += '\n'

    # write complete script
    complete_script_path = script_path + 'setting.sh'
    fobj = open(complete_script_path, 'w')
    fobj.write(complete_run_text)
    fobj.close()
    # execute bit
    os.system('chmod +x ' + complete_script_path)

    os.system(complete_script_path)


if __name__ == '__main__':
    raise SystemExit(main())
