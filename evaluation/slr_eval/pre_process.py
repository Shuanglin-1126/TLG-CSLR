import re
import os
modes = ['train', 'dev', 'test']

for mode in modes:
    tmp_stm_path_read = f"/data/che_xiao/my_project/AdaptSign-main/evaluation/slr_eval_ori/CSL-Daily-groundtruth-{mode}.stm"
    tmp_stm_path_w = f"/data/che_xiao/my_project/AdaptSign-main/evaluation/slr_eval/CSL-Daily-groundtruth-{mode}.stm"
    with open(tmp_stm_path_read, 'r') as f:
        lines = f.readlines()

    with open(tmp_stm_path_w, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                end_time = parts[4]
                # Replace any suspicious large number or inf
                if '1.79769' in end_time or 'inf' in end_time or 'e+' in end_time.lower():
                    parts[4] = '999.0'
                f.write(' '.join(parts) + '\n')
            else:
                f.write(line)


# sys_file = r"/data/che_xiao/my_project/AdaptSign-main/output/phoenix/vitb16_lora/out.output-hypothesis-dev-conv.ctm.sys"
#
# if not os.path.exists(sys_file):
#     raise FileNotFoundError(f"sclite did not generate {sys_file}. Check input format.")
#
# with open(sys_file, 'r') as f:
#     content = f.read()


# with open(sys_file, 'r') as f:
#     for line in f:
#         line = line.strip()
#         if line.startswith('| Sum/Avg'):
#             numbers = re.findall(r'[\d.]+', line)
#             if len(numbers) >= 8:
#                 total_words = float(numbers[0])
#                 total_errors = float(numbers[6])
#                 wer = total_errors / total_words * 100
#                 print(total_words, total_errors, wer)
#                 break

