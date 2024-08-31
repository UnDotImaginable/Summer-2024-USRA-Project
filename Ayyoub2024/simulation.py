from FunctionDefinitions.callAPIFunc import getWeatherData
from FunctionDefinitions.simulationOrganization import simulationProcess
import FunctionDefinitions.colorCoded as color
import FunctionDefinitions.regionComparison as region

# State your intentions
GatherDataFromWebAPI = False  # Step 1 for both ANN and Simulation
ProcessDataForInteractiveTool = True  # Step 2 for Simulation
RunColorCodedSimulation = True  # Step 3 for Simulation
RunRegionComparisonSimulation = False  # Step 3 for Simulation

# Calling API Function, input params to change
output_filename = 'NorthRegion2017_2021.csv'
location = '41.50561, -81.67557'  # Example location (latitude, longitude) CAN ALSO BE IN ADDRESS FORMAT
start_year = 2017
end_year = 2021
if GatherDataFromWebAPI: getWeatherData(output_filename, location, start_year, end_year)

# Calling simulation processing function
folder = 'OrganizedForSim/DowntownRegion/'  # Specify the region that way you can use it for region comparison if you want to use a different region
readFile = output_filename # The file you want to read
data_types = ["Wind Gust",
              "Solar Radiation",
              "Wind Speed",
              "Temperature",
              "Relative Humidity",
              "Wind Direction",
              "Solar Energy",
              "Precipitation",
              "Wind Power Density",
              "Wind + Solar Energy"]
if ProcessDataForInteractiveTool: simulationProcess(readFile, data_types, folder)

# Run Color Coded Simulation
if RunColorCodedSimulation: color.create_dashboard(folder, data_types)

# Run region comparison simulation
parentFolder = 'OrganizedForSim/'
regions = ['DowntownRegion'] # add more regions to the array if you have collected more data
if RunRegionComparisonSimulation: region.create_dashboard(parentFolder, regions)
