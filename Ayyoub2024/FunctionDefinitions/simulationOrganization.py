import pandas as pd
import numpy as np
import csv
from datetime import datetime, timedelta
import os

def process_data(data, dataType, output_folder):
    if isinstance(data, str):
        newDataFrame = pd.read_csv(data, low_memory=False)
    else:
        newDataFrame = data

    DateTime = newDataFrame['Date time'].to_numpy()
    WantData = newDataFrame[dataType].to_numpy()

    WantData = pd.to_numeric(WantData, errors='coerce')   

    ogLen = len(WantData)
    indices = np.where(np.isnan(WantData))
    WantData = np.delete(WantData, indices)
    DateTime = np.delete(DateTime, indices)
    numBad = len(indices[0])

    percentGood = 1 - (numBad / ogLen)

    if dataType == 'Wind Speed':
        WantData = WantData / 3.6
        #write percent of valid windspeed data for documentation purposes
        #with open(os.path.join(output_folder, 'percentGood.txt'), 'w') as f:
        #    f.write(str(data) + '\n')
        #    f.write(str(percentGood) + '\n')

    return WantData, DateTime, percentGood

def getCurrentTimeIndex(dateTime):
    currentTimeIndex = dateTime.split('/')[2][5:10].split(':')
    currentTimeIndex = int(currentTimeIndex[0]) * 6 + int(currentTimeIndex[1]) // 10
    return currentTimeIndex

def generalAvgByHour(WS, DT):
    i, j, y = 1, 0, -1
    currentTimeIndex = getCurrentTimeIndex(DT[0])
    currentDay = DT[0].split('/')[0] + '/' + DT[0].split('/')[1] + '/' + DT[0].split('/')[2][0:4]
    nextDay = currentDay

    rows, cols = (len(DT), 6 * 24 + 1)
    hourData = [[''] * cols for _ in range(rows)]
    while j < len(DT):
        currentDay = DT[j].split('/')[0] + '/' + DT[j].split('/')[1] + '/' + DT[j].split('/')[2][0:4]
        currentTimeIndex = getCurrentTimeIndex(DT[j])
        if dayToNum(currentDay) >= dayToNum(nextDay):
            nextDay = increment_day(nextDay)
            i += 1
        if hourData[0][currentTimeIndex] == '':
            hourData[0][currentTimeIndex] = DT[j].split('/')[2][5:10]
        hourData[i][currentTimeIndex] = WS[j]
        j += 1
    #print('Done running split by generalHour')
    return hourData

def splitsByYear(WS, DT, numberOfYears):
    i, j, y = 0, 0, -1
    currentYear = DT[0].split('/')[2][0:4]
    nextYear = currentYear

    rows, cols = (len(DT), numberOfYears + 1)
    yearData = [[''] * cols for _ in range(rows)]
    while j < len(DT):
        currentYear = DT[j].split('/')[2][0:4]
        if currentYear == nextYear:
            y += 1
            yearData[0][y] = nextYear
            nextYear = str(int(nextYear) + 1)
            i = 1

        yearData[i][y] = WS[j]
        i += 1
        j += 1
    yearData = getRidOfEmptyRows(yearData)
    #print('Done running split by year')
    return yearData

def splitsByMonth(WS, DT, numOfYears):
    i, j, m = 0, 0, -1
    currentMonth = DT[0].split('/')[0] + '/' + DT[0].split('/')[2][0:4]
    nextMonth = currentMonth

    rows, cols = (len(DT) // numOfYears, (13 * numOfYears))
    monthData = [[''] * cols for _ in range(rows)]

    while j < len(DT):
        currentMonth = DT[j].split('/')[0] + '/' + DT[j].split('/')[2][0:4]
        if currentMonth == nextMonth:
            m += 1
            monthData[0][m] = nextMonth
            nextMonth = increment_month_with_zero(nextMonth)
            i = 1

        monthData[i][m] = WS[j]
        i += 1
        j += 1
    monthData = getRidOfEmptyRows(monthData)
    #print('Done running split by month')
    return monthData

def splitsByDays(WS, DT, numOfYears):
    i, j, d = 0, 0, -1
    currentDay = DT[0].split('/')[0] + '/' + DT[0].split('/')[1] + '/' + DT[0].split('/')[2][0:4]
    nextDay = currentDay

    rows, cols = (24 * 7, 13 * numOfYears * 31)
    dayData = [[''] * cols for _ in range(rows)]

    while j < len(DT):
        currentDay = DT[j].split('/')[0] + '/' + DT[j].split('/')[1] + '/' + DT[j].split('/')[2][0:4]
        if dayToNum(currentDay) >= dayToNum(nextDay):
            d += 1
            dayData[0][d] = currentDay
            nextDay = increment_day(nextDay)
            i = 1
        dayData[i][d] = WS[j]
        i += 1
        j += 1
    dayData = getRidOfEmptyRows(dayData)
    #print('Done running split by days')
    return dayData

def increment_month_with_zero(date_str):
    m, y = map(int, date_str.split('/'))
    if m == 12:
        m = 1
        y += 1
    else:
        m += 1
    return f"{m}/{y}" if m >= 10 else f"0{m}/{y}"

def splitsDayByYear(data, numberOfYears):
    days, data = data[0], data[1]
    rows, cols = (len(data), numberOfYears + 1)
    yearData = [[np.nan] * cols for _ in range(rows)]
    currentYear = days[0].split('/')[2]
    nextYear = currentYear

    yearIndex, j = 0, 1
    yearData[0][yearIndex] = currentYear
    data = np.append(data, [np.nan] * (len(days) - len(data)))
    for day in range(len(days) - 1):
        if days[day] == '':
            break
        currentYear = days[day].split('/')[2]
        if currentYear != nextYear:
            nextYear = str(int(nextYear) + 1)
            yearIndex += 1
            yearData[0][yearIndex] = currentYear
            j = 1
        yearData[j][yearIndex] = data[day]
        j += 1
    return yearData

def splitsDayByMonth(data, numberOfYears):
    days, data = data[0], data[1]
    rows, cols = (len(data) // numberOfYears, (numberOfYears + 1) * 12)
    monthData = [[np.nan] * cols for _ in range(rows)]
    currentMonth = days[0].split('/')[0] + '/' + days[0].split('/')[2]
    nextMonth = currentMonth

    monthIndex, j = 0, 1
    monthData[0][monthIndex] = currentMonth
    data = np.append(data, [np.nan] * (len(days) - len(data)))
    for day in range(len(days) - 1):
        if days[day] == '':
            break
        currentMonth = days[day].split('/')[0] + '/' + days[day].split('/')[2]
        if currentMonth != nextMonth:
            nextMonth = increment_month_with_zero(nextMonth)
            monthIndex += 1
            monthData[0][monthIndex] = currentMonth
            j = 1
        monthData[j][monthIndex] = data[day]
        j += 1
    return monthData

def increment_day(date_str):
    m, d, y = map(int, date_str.split('/'))
    date = datetime(year=y, month=m, day=d)
    next_day = date + timedelta(days=1)
    formatted_date = f"{next_day.month}/{next_day.day}/{next_day.year}"
    return formatted_date

def dayToNum(date_str):
    m, d, y = map(int, date_str.split('/'))
    return datetime(year=y, month=m, day=d)

def takeAverages(data, expectedLength):
    rows, cols = (2, len(data[0]))
    avgs = [[np.nan] * cols for _ in range(rows)]
    avgs[0] = data[0]
    data = np.delete(data, 0, 0)
    data = np.where(data == '', np.nan, data).astype(np.double)

    raw_averages = []
    daysRemoved = 0
    for i in range(len(data[0])):
        column_data = data[:, i]
        non_nan_data = column_data[~np.isnan(column_data)]

        if len(non_nan_data) < expectedLength:
            raw_averages.append(np.nan)
            daysRemoved += 1
            continue

        sum_non_nan = np.nansum(non_nan_data)
        if sum_non_nan == 0 or np.isnan(sum_non_nan):
            raw_averages.append(sum_non_nan)
            continue

        nnn = len(non_nan_data)
        avg = sum_non_nan / nnn
        raw_averages.append(avg)

    raw_averages = np.array(raw_averages)
    avgs[1] = raw_averages
    #print('Elements removed: ', daysRemoved)
    #print('Done running takeAverages')
    return avgs

def getRidOfEmptyRows(data):
    final_ind = 0
    data = np.array(data)
    for i in range(len(data[0])):
        indices_to_delete = np.where(data[:, i] == '')[0]

        if len(indices_to_delete) > 0:
            final_indT = indices_to_delete[0]

            if final_indT > final_ind:
                final_ind = final_indT

    #print('final index is: ', final_ind)
    return data[:final_ind, :]

def convertToWindPowerDensity(windSpeed, rho=1.225):
    windEnergy = 0.5 * rho * windSpeed**3
    return windEnergy, 'Wind Power Density'

def readFromGenerated(filename):
    if os.path.isfile(filename) and os.access(filename, os.R_OK):
        newDataFrame = pd.read_csv(filename, header=None)
        DateTime = newDataFrame.iloc[0].values.tolist()
        WantData = newDataFrame.iloc[1].values.tolist()
        return WantData, DateTime
    else:
        print(f"Cannot read file: {filename}")
        return [], []

def simulationProcess(data, data_types, output_folder, num_years):
    if isinstance(data, str):
        readFileName = data
    else:
        readFileName = 'data_from_dataframe.csv'
        data.to_csv(readFileName, index=False)

    for dataType in data_types:
        print('Beginning Organization for: ', dataType)
        oldDataType = ''
        if dataType == 'Wind Power Density':
            dataType = 'Wind Speed'
            oldDataType = 'Wind Power Density'
        if dataType != 'Wind + Solar Energy':
            WantData, DateTime, percentGood = process_data(readFileName, dataType, output_folder)
        if oldDataType == 'Wind Power Density':
            WantData, dataType = convertToWindPowerDensity(WantData)
        if dataType == 'Wind + Solar Energy':
            SolarData, SolarDateTime = readFromGenerated(os.path.join(output_folder, 'Solar Radiation/daily.csv'))
            WindData, WindDateTime = readFromGenerated(os.path.join(output_folder, 'Wind Power Density/daily.csv'))
            if(WindData == [] or SolarData == []): continue
            WindSolar = []
            WindSolarDT = []
            for i in range(len(SolarDateTime)):
                if SolarDateTime[i] in WindDateTime:
                    WindSolar.append(float(WindData[WindDateTime.index(SolarDateTime[i])]) + float(SolarData[i]))
                    WindSolarDT.append(SolarDateTime[i])
            dayData = [WindSolarDT, WindSolar]

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        wType = os.path.join(output_folder, dataType)
        if not os.path.exists(wType):
            os.mkdir(wType)

        if dataType != 'Wind + Solar Energy':
            if WantData.size == 0:
                print('No data for this type: ', dataType)
                continue
            elif WantData.size < 6*24*30:
                print('Not enough data for this type: ', dataType)
                continue
            dayData = splitsByDays(WantData, DateTime, num_years)
            expectedLength = 6 * 24 * percentGood * .75
            dayData = takeAverages(dayData, expectedLength)
        yearData = splitsDayByYear(dayData, num_years)
        expectedLength = 1
        yearData = takeAverages(yearData, expectedLength)
        csv_file = os.path.join(wType, 'yearly.csv')
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(yearData)
        print('Done writing year data')

        monthData = splitsDayByMonth(dayData, num_years)
        expectedLength = 1
        monthData = takeAverages(monthData, expectedLength)
        csv_file = os.path.join(wType, 'monthly.csv')
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(monthData)
        print('Done writing month data')

        csv_file = os.path.join(wType, 'daily.csv')
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(dayData)
        print('Done writing day data')

        if dataType == 'Wind + Solar Energy':
            SolarData, SolarDateTime = readFromGenerated(os.path.join(output_folder, 'Solar Radiation/hourly.csv'))
            WindData, WindDateTime = readFromGenerated(os.path.join(output_folder, 'Wind Power Density/hourly.csv'))
            if(WindData == [] or SolarData == []): continue
            WindSolar = []
            WindSolarDT = []
            for i in range(len(SolarDateTime)):
                if SolarDateTime[i] in WindDateTime:
                    WindSolar.append(float(WindData[WindDateTime.index(SolarDateTime[i])]) + float(SolarData[i]))
                    WindSolarDT.append(SolarDateTime[i])
            hourData = [WindSolarDT, WindSolar]
        else:
            hourData = generalAvgByHour(WantData, DateTime)
        hourData = takeAverages(hourData, 1)
        csv_file = os.path.join(wType, 'hourly.csv')
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(hourData)
        print('Done writing hour data')

        print('Done with organizing data for: ', dataType)
        print('\n')
    print('Done with all iterations')
    return output_folder
