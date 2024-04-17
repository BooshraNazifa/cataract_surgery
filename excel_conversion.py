import pandas as pd
import numpy as np

test_results_df = pd.read_excel('./test_results.xlsx')
test_results_df_sorted = test_results_df.sort_values(by=['Filename', 'Timestamp'])

# Group by 'Filename' and 'True Phase' to get the start and end times for each phase in each video
grouped = test_results_df_sorted.groupby(['Filename', 'Predicted Phase'])

# Create a new DataFrame for the phase times
phase_times_list = []

# Iterate over each group and extract the start and end time for each phase
for (filename, true_phase), group in grouped:
    start_time = group['Timestamp'].min()
    end_time = group['Timestamp'].max()
    phase_times_list.append({'Filename': filename, 'Phase': true_phase, 'Start': start_time, 'End': end_time})

# Convert to DataFrame
phase_times_df = pd.DataFrame(phase_times_list)

# Convert 'Start' and 'End' times from seconds to the format 'HH:MM:SS:FF'
def seconds_to_hmsf(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    frames = int((seconds % 1) * 100)  # Assuming 100 frames per second for calculation
    return f"{hours:02}:{minutes:02}:{seconds:02}:{frames:02}"

# Apply the conversion function to 'Start' and 'End' columns
phase_times_df['Start'] = phase_times_df['Start'].apply(seconds_to_hmsf)
phase_times_df['End'] = phase_times_df['End'].apply(seconds_to_hmsf)

# Now we pivot the DataFrame to the desired shape
pivot_df = phase_times_df.pivot(index='Phase', columns='Filename', values=['Start', 'End'])

# Reset the index to make 'Phase' a column again
pivot_df.reset_index(inplace=True)
pivot_df.columns = [' '.join(col).strip() if col[0] else col[1] for col in pivot_df.columns.values]

# Now we order the columns as per the desired output
ordered_columns = ['Phase'] + [col for col in pivot_df.columns if col != 'Phase']
pivot_df = pivot_df[ordered_columns]
current_dict = pivot_df.to_dict()
print(current_dict)

files = set(k.split()[1] for k in current_dict.keys() if k != 'Phase')

final_dict = {
    '': {
        0: 'Phase',
        1: 'Paracentesis',
        2: 'Viscoelastic',
        3: 'Wound',
        4: 'Capsulorhexis',
        5: 'Hydrodissection',
        6: 'Phaco',
        7: 'Viscoelastic2',
        8: 'IOL Insertion',
        9: 'IOL Positioning',
        10: 'Viscoelastic removed',
        11: 'Hydration',
        12: 'Malyugin Ring Insertion',
        13: 'Malyugin Ring Removal',
        14: 'Vision Blue'
    }
}

for file in files:
    start_key = f'Start {file}'
    end_key = f'End {file}'

    # Initialize 'Start' dictionary entry with 'Start' label at index 0
    final_dict[f'{file} Start'] = {0: 'Start'}
    final_dict[f'{file} End'] = {0: 'End'}
    
    # Fill in start times, shifting indices by 1 to accommodate the 'Start' label
    for i in range(len(current_dict['Phase'])):
        final_dict[f'{file} Start'][i + 1] = current_dict[start_key][i]
        final_dict[f'{file} End'][i + 1] = current_dict[end_key][i]



print(final_dict)


results_df = pd.DataFrame(final_dict)
print(results_df)
results_df.to_excel('test_results_by_phase.xlsx', index=False)