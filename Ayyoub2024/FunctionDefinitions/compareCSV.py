import pandas as pd
import numpy as np

# Load the CSV files
folder = '16_cleveland_test/'
filename = folder + 'W2.csv'  # Specify your CSV file name that contains the downloaded data
filename2 =folder + 'W3.csv'

# Read the data
data = pd.read_csv(filename, low_memory=False)
data2 = pd.read_csv(filename2, low_memory=False)

# Column names
date_column_name = 'Date time'  # Update with your actual column name
wind_column_name = 'Wind Speed'  # Update with your actual column name

#Process first
DateTime = data['Date time'].to_numpy()
WindData = data[wind_column_name].to_numpy()
WindData = pd.to_numeric(WindData, errors='coerce')

indices = np.where(WindData == 0)
WindData = np.delete(WindData, indices)
DateTime = np.delete(DateTime, indices)
indices = np.where(np.isnan(WindData))
WindData = np.delete(WindData, indices)
DateTime = np.delete(DateTime, indices)

#Process second
DateTime2 = data2['Date time'].to_numpy()
WindData2 = data2[wind_column_name].to_numpy()
WindData2 = pd.to_numeric(WindData2, errors='coerce')

indices = np.where(WindData2 == 0)
WindData2 = np.delete(WindData2, indices)
DateTime2 = np.delete(DateTime2, indices)
indices = np.where(np.isnan(WindData2))
WindData2 = np.delete(WindData2, indices)
DateTime2 = np.delete(DateTime2, indices)


# Convert average wind speed from km/h to m/s
WindData = WindData * 1000 / 3600
WindData2 = WindData2 * 1000 / 3600


# Convert to DataFrame for merging
df1 = pd.DataFrame({'DateTime': DateTime, 'WindData': WindData})
df2 = pd.DataFrame({'DateTime': DateTime2, 'WindData2': WindData2})

# Merge dataframes on DateTime
merged_df = pd.merge(df1, df2, on='DateTime')

# Extract the intersecting wind data
WindData = merged_df['WindData'].to_numpy()
WindData2 = merged_df['WindData2'].to_numpy()



print('percent differnce')
print((np.mean(abs(WindData - WindData2))/((np.mean(WindData+WindData2))/2))*100)

