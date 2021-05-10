import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
from numerize import numerize
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import interact, widgets


def cleanData(dataSet: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to clean the data and make it consistent for further processing.
    :param dataSet: Dataframe to be cleaned
    :return: Dataframe with appropriate transformations.

    >>> stateElections1 = pd.DataFrame(np.array([['16/01/2021','Andaman and Nicobar Islands'],['17/01/2021','Andaman and Nicobar Islands'],['16/01/2021','Delhi'],['17/01/2021','Delhi']]), columns = ['Updated On','State'])
    >>> cleanData(stateElections1)
       Updated On                      State
    0  16/01/2021  Andaman & Nicobar Islands
    1  17/01/2021  Andaman & Nicobar Islands
    2  16/01/2021               NCT OF Delhi
    3  17/01/2021               NCT OF Delhi

    >>> stateElections2 = pd.DataFrame(np.array([['16/01/2021','2'],['17/01/2021','2'],['16/01/2021','91'],['17/01/2021','288']]), columns = ['Updated On','Total Sessions Conducted'])
    >>> cleanData(stateElections2)
    Traceback (most recent call last):
    KeyError: 'State'

    """
    dataSet_copy = dataSet.copy()
    mapper = pd.read_csv('mapper.csv')
    dataSet_copy['State'].replace(to_replace=mapper.to_replace.tolist(),
                                  value=mapper.value.tolist(), inplace=True)

    return dataSet_copy


def getStateGovernments(stateElect: pd.DataFrame) -> pd.DataFrame:
    """
    This function aims to find the Ruling Party in each state and the votes it received.
    :param stateElect: Dataframe with election data
    :return: Dataframe with states and the corresponding political party in power.

    >>> getStateGovernments1 = pd.DataFrame(np.array([['Manipur','Inner manipur','263632','Bharatiya Janata Party','DR RAJKUMAR RANJAN SINGH'],['Manipur','Outer manipur','363527','Naga Peoples Front','Lorho S. Pfoze'],['Meghalaya','Shillong','419689','Indian National Congress','VINCENT H. PALA'],['Meghalaya','Tura','304455','National Peoples Party','AGATHA K. SANGMA']]),columns = ['State','Constituency','Votes','Party','Candidate'])
    >>> getStateGovernments(getStateGovernments1)
           State                     Party   Votes
    2  Meghalaya  Indian National Congress  419689
    1    Manipur        Naga Peoples Front  363527

    >>> getStateGovernments2 = pd.DataFrame(np.array([['Manipur','Inner manipur','Bharatiya Janata Party','DR RAJKUMAR RANJAN SINGH'],['Manipur','Outer manipur','Naga Peoples Front','Lorho S. Pfoze'],['Meghalaya','Shillong','Indian National Congress','VINCENT H. PALA'],['Meghalaya','Tura','National Peoples Party','AGATHA K. SANGMA']]),columns = ['State','Constituency','Party','Candidate'])
    >>> getStateGovernments(getStateGovernments2)
    Traceback (most recent call last):
    pandas.core.base.SpecificationError: Column(s) ['Votes'] do not exist
    """
    groupByStateParty = stateElect.groupby(['State', 'Party']).agg({'Votes': 'sum'})
    groupByStateParty.columns = ['Votes']
    groupByStateParty = groupByStateParty.reset_index()

    stateRulingParty = groupByStateParty.sort_values(['Votes'], ascending=False)
    stateRulingParty = stateRulingParty.drop_duplicates(subset=['State'], keep='first')

    return stateRulingParty


def getStateVaccineRecords(stateVaccines: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to find the vaccine data for each state based on the latest updated date.
    :param stateVaccines: Dataframe with vaccine data statewise.
    :return: Dataframe with sates and corresponding number of individuals vaccinated as per latest updated date.

    >>> getStateVaccineRecords1 = pd.DataFrame(np.array([['20/04/2021','Manipur','109878'],['21/04/2021','Manipur','114345'],['20/04/2021','Sikkim','146868'],['21/04/2021','Sikkim','147787']]),columns = ['Updated On','State','Total Individuals Registered'])
    >>> getStateVaccineRecords(getStateVaccineRecords1)
       index Updated On    State Total Individuals Registered
    0      1 2021-04-21  Manipur                       114345
    1      3 2021-04-21   Sikkim                       147787

    >>> getStateVaccineRecords2 = pd.DataFrame(np.array([['Manipur','109878'],['Manipur','114345'],['Sikkim','146868'],['Sikkim','147787']]),columns = ['State','Total Individuals Registered'])
    >>> getStateVaccineRecords(getStateVaccineRecords2)
    Traceback (most recent call last):
    KeyError: 'Updated On'
    """
    stateVaccines = cleanData(stateVaccines)
    stateVaccines = stateVaccines.loc[stateVaccines['State'] != 'India']
    stateVaccines['Updated On'] = pd.to_datetime(stateVaccines['Updated On'], infer_datetime_format=True)
    latestVaccineInfo = stateVaccines.sort_values(['Updated On'], ascending=False)
    latestVaccineInfo = latestVaccineInfo.drop_duplicates(subset=['State'], keep='first')
    latestVaccineInfo = latestVaccineInfo.reset_index()
    return latestVaccineInfo


def calcVaccinationRate(Total_Individuals_Vaccinated: float, Population_2019: float)-> float:
    """
    The purpose of this function is to calculate the VaccineRate in each state by finding the ratio of Total Individuals Vaccinated and Population.
    :param Total_Individuals_Vaccinated: Number of individuals vaccinated
    :param Population_2019: Population of the state
    :return: VaccineRate(float)

    >>> calcVaccinationRate(3456723,45637289)
    7.574339045424017
    >>> calcVaccinationRate(3456723,0)
    Traceback (most recent call last):
    ZeroDivisionError: division by zero
    >>> calcVaccinationRate(0,0)
    Traceback (most recent call last):
    ZeroDivisionError: division by zero
    """
    VaccinationRate = (Total_Individuals_Vaccinated / Population_2019) * 100
    return VaccinationRate


def calcRateOfVaccination(vaccineRecords: pd.DataFrame, statePop: pd.DataFrame,
                          stateGovernments: pd.DataFrame) -> tuple:
    """
    This function is used to calculate the VaccineRate and add it as a new column in the dataframe.
    :param vaccineRecords: Vaccine data
    :param statePop: Population data
    :param stateGovernments: Election data
    :return: Tuple with two dataframes consisting of vaccination rates.

    >>> stateGovernments1 = pd.DataFrame(np.array([['Manipur','Inner manipur','Bharatiya Janata Party','DR RAJKUMAR RANJAN SINGH'],['Meghalaya','Shillong','Indian National Congress','VINCENT H. PALA']]),columns = ['State','Constituency','Party','Candidate'])
    >>> statPop1 = pd.DataFrame(np.array([['Meghalaya',3366710.0,2966889],['Manipur',3091545.0,2855794]]), columns = ['State','Population_2019','Population_2011'])
    >>> vaccineRecords1 = pd.DataFrame(np.array([['20/04/2021','Manipur','109878','11400','108','109878','52226','73805','36068','5','0','162104',109878.0,162104,],['20/04/2021','Meghalaya','133716','29000','260','133716','46932','68644','65059','13','0','180648',133716.0,180648]]), columns = ['Updated On','State','Total Individuals Registered','Total Sessions Conducted','Total Sites','First Dose Administered','Second Dose Administered','Male(Individuals Vaccinated)','Female(Individuals Vaccinated)','Transgender(Individuals Vaccinated)','Total Covaxin Administered','Total CoviShield Administered','Total Individuals Vaccinated','Total Doses Administered'])
    >>> calcRateOfVaccination(vaccineRecords1,statPop1,stateGovernments1)
    """
    vaccinePopulation = statePop[['State', 'Population_2019']].merge(
                        vaccineRecords[['Updated On', 'State', 'Total Individuals Vaccinated']], on='State')
    vaccinePopulation['VaccinationRate'] = vaccinePopulation.apply(
        lambda row: calcVaccinationRate(row['Total Individuals Vaccinated'], row['Population_2019']), axis=1)

    vaccinePopulation1 = statePop[['State', 'Population_2019']].merge(
                         vaccineRecords[['Updated On', 'State', 'Total Individuals Vaccinated']], on='State').merge(
                         stateGovernments[['State', 'Party']], on='State')

    grouped = vaccinePopulation1.groupby('Party').agg({'Population_2019': 'sum', 'Total Individuals Vaccinated': 'sum'})
    grouped = grouped.reset_index()
    grouped['VaccinationRate'] = grouped.apply(
        lambda row: calcVaccinationRate(row['Total Individuals Vaccinated'], row['Population_2019']), axis=1)

    return (grouped, vaccinePopulation)


def stateClassificationByRulingParty(stateGovernments: pd.DataFrame, rateOfVaccination: pd.DataFrame) -> pd.DataFrame:
    """
    This function aims at merging the election dataframe and VaccineRate dataframe for further processing.
    :param stateGovernments: Election data
    :param rateOfVaccination: VaccineRate data
    :return: Dataframe with states classified based on the currently ruling party.
    """
    vaccinationRate_by_rulingParty = stateGovernments[['State', 'Party']].merge(
        rateOfVaccination[['State', 'Population_2019', 'Total Individuals Vaccinated', 'VaccinationRate']], on='State')
    return vaccinationRate_by_rulingParty


def hypothesis1(electionData, vaccineData, populationData):
    """
    This function consists of all the steps required for Hypothesis 1.
    All the function calls for Hypothesis 1 are made from this function.
    :param electionData: Dataframe with results for 2019 elections in India.
    :param vaccineData: Dataframe with vaccination records for each state in India.
    :param populationData: Dataframe with population in each state as of 2019.
    """

    # CLeaning the data to have uniform names of states across the input files.
    statePopulation = cleanData(populationData)

    # Fetching the currently ruling parties in each state of India.
    stateGovernments = getStateGovernments(electionData)

    # Fetching the statewise vaccination records for the latest dates.
    stateVaccineRecords = getStateVaccineRecords(vaccineData)

    # Calculating the rate of vaccination in each state.
    rateOfVaccination_vs_party = calcRateOfVaccination(stateVaccineRecords, statePopulation, stateGovernments)

    # Plotting bar plot for States within each party vs vaccination rate.
    drp_dwn = widgets.Dropdown(value='Total Individuals Vaccinated',
                               options=[('# vaccinated', 'Total Individuals Vaccinated'),
                                        ('% vaccinated', 'VaccinationRate')])

    @interact(y=drp_dwn)
    def plot(y):
        fig = px.bar(rateOfVaccination_vs_party[0], x='Party', y=y, log_y=True, title='Vaccination per Political Party',
                     hover_name='Party',
                     hover_data={'a': [numerize.numerize(x, 0) for x in
                                       rateOfVaccination_vs_party[0].Population_2019.tolist()],
                                 'tiv': [numerize.numerize(x, 0) for x in
                                         rateOfVaccination_vs_party[0]['Total Individuals Vaccinated'].tolist()],
                                 'vr': [round(x, 2) for x in
                                        rateOfVaccination_vs_party[0]['VaccinationRate'].tolist()],
                                 y: False,
                                 'VaccinationRate': False,
                                 'Party': False,
                                 'Population_2019': False},
                     labels={'a': 'population',
                             'tiv': '# vaccinated',
                             'vr': '% vaccinated'},
                     color='Population_2019',
                     color_continuous_scale='darkmint')
        fig.update_layout(plot_bgcolor='white')
        fig.show()

    # Classifying the states by ruling parties.
    vaccinationRate_by_rulingParty = stateClassificationByRulingParty(stateGovernments, rateOfVaccination_vs_party[1])

    # Plotting bar graph for States vs Vaccination rate and coloring the bar plot to represent the ruling party.
    fig, ax = plt.subplots(figsize=(100, 30))
    for key, grp in vaccinationRate_by_rulingParty.groupby(['Party']):
        ax.bar(grp['State'], grp['VaccinationRate'], label=key)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(55)
    ax.legend(loc=2, prop={'size': 40})
    ax.tick_params(axis='x', labelrotation=90)
    plt.title(label="States Vs Vaccination Rates",
              fontsize=95)
    plt.xlabel('States', fontsize=55)
    plt.ylabel('Vaccination Rates', fontsize=55)
    plt.show()


def getTestGrouped(testData: pd.DataFrame) -> pd.DataFrame:
    """
    Combines number of tests conducted and positive cases for each day recorded in dataset for every state.
    :param testData: Data frame consisting of the COVID-19 testing data for each state.
    :return: Creates a dataframe having total number of tests conducted and positive cases for each day recorded in
    India.
    """
    testingTotal = testData.groupby(['Updated On']).agg({'Total Tested': 'sum'})
    testingTotal = testingTotal.reset_index()

    testingPositive = testData.groupby(['Updated On']).agg({'Positive': 'sum'})
    testingPositive = testingPositive.reset_index()

    testGrouped = testingTotal.merge(testingPositive, on='Updated On')

    return testGrouped


def getTotalDailyVaccinated(vaccination: pd.DataFrame) -> pd.DataFrame:
    """
    Combines number of individuals vaccinated for each day across all the states.
    :param vaccination: Dataframe consisting of vaccination records for each state.
    :return: Dataframe having total number of individuals vaccinated for each day recorded.
    """
    vaccination_ = vaccination.copy()
    vaccination_['Updated On'] = pd.to_datetime(vaccination_['Updated On'], infer_datetime_format=True)
    vaccineCombined = vaccination_.groupby(['Updated On']).agg({'Total Individuals Vaccinated': 'sum'})
    vaccineCombined = vaccineCombined.reset_index()

    return vaccineCombined


def plotHypo2(testingVsVaccinated: pd.DataFrame, title: str):
    """
    This functions creates all the three plots required for the hypothesis 2.
    :param testingVsVaccinated: Dataframe consisting of testing & vaccination details for each date across different states.
    :param title: String which is used as title of the plots.
    """
    fig = go.Figure()
    if "before" in title:
        fig.add_trace(go.Scatter(x=testingVsVaccinated['Updated On'], y=testingVsVaccinated['Total Tested'],
                                 mode='lines',
                                 name='Total Tests Conducted'))

        fig.add_trace(go.Scatter(x=testingVsVaccinated['Updated On'], y=testingVsVaccinated['Positive'],
                                 mode='lines',
                                 name='Total Positive Cases'))

    else:
        fig.add_trace(go.Scatter(x=testingVsVaccinated['Updated On'], y=testingVsVaccinated['Total Tested'],
                                 mode='lines',
                                 name='Total Tests Conducted'))

        fig.add_trace(
            go.Scatter(x=testingVsVaccinated['Updated On'], y=testingVsVaccinated['Total Individuals Vaccinated'],
                       mode='lines',
                       name='Total Individuals Vaccinated'))

        fig.add_trace(go.Scatter(x=testingVsVaccinated['Updated On'], y=testingVsVaccinated['Positive'],
                                 mode='lines',
                                 name='Total Positive Cases'))
    fig.update_layout(
        title={
            'text': title,
            'y': 0.9,
            'x': 0.4,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Time",
        yaxis_title="Rates",
    )

    fig.show()


def hypothesis2(testingData: pd.DataFrame, vaccinesData: pd.DataFrame):
    """
    This function consists of all the steps required for Hypothesis 1.
    All the function calls for Hypothesis 1 are made from this function.
    :param testingData: Data frame consisting of the COVID-19 testing data for each state.
    :param vaccinesData: Dataframe consisting of vaccination records for each state.
    """

    print("#########################################################################################################")
    print("                                            HYPOTHESIS 2                                                 ")
    print("#########################################################################################################")

    testingData['Updated On'] = pd.to_datetime(testingData['Updated On'], infer_datetime_format=True)

    # Lets first check the trend in COVID-19 testing before vaccinations began.
    testData = testingData.drop(testingData[testingData['Updated On'].dt.year == 2021].index)
    testbeforevaccine = getTestGrouped(testData)
    plotHypo2(testbeforevaccine, "Testing conducted before vaccinations")

    testingData.drop(testingData[testingData['Updated On'].dt.year != 2021].index, inplace = True)
    # Fetching total number of tests conducted each day across different states.
    testingGrouped = getTestGrouped(testingData)
    # Fetching total number individuals vaccinated each day across different states.
    vaccineGrouped = getTotalDailyVaccinated(vaccinesData)

    # Merging the testing and vaccination details for creating plot for overall tests conducted after vaccination began.
    testingVsVaccinated = vaccineGrouped.merge(testingGrouped, on='Updated On')

    # Fetching the records for 1st shot of vaccinations only.
    vaccine_1st_shot = vaccinesData.loc[vaccinesData['Second Dose Administered'] == 0]
    # Fetching the records for 2nd shot of vaccinations only.
    vaccine_2nd_shot = vaccinesData.loc[vaccinesData['Second Dose Administered'] != 0]

    # Fetching total number individuals who got first shot each day across different states.
    vaccine_1st_grouped = getTotalDailyVaccinated(vaccine_1st_shot)
    # Fetching total number individuals who got second shot each day across different states.
    vaccine_2nd_grouped = getTotalDailyVaccinated(vaccine_2nd_shot)

    first_shot_date = vaccine_1st_grouped['Updated On'].max()
    second_shot_date = vaccine_2nd_grouped['Updated On'].min()

    # Fetching the records for tests conducted after 1st shot and before the second shots started.
    test_after_1st_shot = testingGrouped.loc[testingGrouped['Updated On'] <= first_shot_date]
    # Fetching the records for tests conducted after 2nd shots were started.
    test_after_2nd_shot = testingGrouped.loc[testingGrouped['Updated On'] >= second_shot_date]

    # Merging the testing & vaccination details for creating a plot for test conducted after first shot of vaccination.
    testingVsVaccinated_1st = vaccine_1st_grouped.merge(test_after_1st_shot, on = 'Updated On')
    # Merging the testing $ vaccination details for creating a plot for test conducted after second shot of vaccination.
    testingVsVaccinated_2nd = vaccine_2nd_grouped.merge(test_after_2nd_shot, on = 'Updated On')

    # Creating all three plots.
    plotHypo2(testingVsVaccinated, "No. of individuals vaccinated Vs Tests Conducted")
    plotHypo2(testingVsVaccinated_1st, "No. of individuals given first shot Vs Tests Conducted after first shot")
    plotHypo2(testingVsVaccinated_2nd, "No. of individuals given second shot Vs Tests Conducted after second shot")


def getVaccineTypeCount (vaccinesTypeData : pd.DataFrame):
    """
    This function calculates the number of shots administered for each type of vaccine in each state.
    :param vaccinesTypeData: Dataframe consisting of data for each type of vaccine
    :return: Dataframe with number of shots administered per vaccine in each state.

    >>> vaccinesTypeData1 = pd.DataFrame(np.array([['20/04/2021','Manipur',10988,11400,108,109878,52226,73805,36068,5,0,162104,109878.0,162104,],['20/04/2021','Meghalaya',133716,29000,260,133716,46932,68644,65059,13,0,180648,133716.0,180648]]), columns = ['Updated On','State','Total Individuals Registered','Total Sessions Conducted','Total Sites','First Dose Administered','Second Dose Administered','Male(Individuals Vaccinated)','Female(Individuals Vaccinated)','Transgender(Individuals Vaccinated)','Total Covaxin Administered','Total CoviShield Administered','Total Individuals Vaccinated','Total Doses Administered'])
    >>> getVaccineTypeCount(vaccinesTypeData1)
           State Total Covaxin Administered Total CoviShield Administered
    0    Manipur                          0                        162104
    1  Meghalaya                          0                        180648
    """
    vaccinesTypeData = vaccinesTypeData.loc[vaccinesTypeData['State'] != 'India']
    vaccineTypeData = vaccinesTypeData.copy()
    groupedVaccineTypeData = vaccineTypeData.groupby('State').agg({'Total Covaxin Administered': 'sum', 'Total CoviShield Administered': 'sum'})
    groupedVaccineTypeData = groupedVaccineTypeData.reset_index()
    return groupedVaccineTypeData


def rank_(grp, col, return_rank=None):
    """
    This function is used to find the rank of the state based on daily distribution of each type of vaccine.
    :param grp:
    :param col:
    :param return_rank:
    :return:

    >>> df = pd.DataFrame(np.array([['2021-01-16', 'Delhi', 23.0, 0.0, 23.0], ['2021-01-16', 'Punjab', 25.0, 1.0, 24.0],
    ... ['2021-01-16', 'Gujrat', 35.0, 6.0, 29.0], ['2021-01-16', 'Maharashtra', 40.0, 6.0, 34.0]]),
    ... columns = ['Updated On','State', 'Total Individuals Vaccinated', 'Total Covaxin Administered',
    ...            'Total CoviShield Administered'])
    >>> rank_(df, 'Total CoviShield Administered', return_rank=None)
             State
    3  Maharashtra
    2       Gujrat
    1       Punjab
    0        Delhi
    """
    if return_rank==None:
        return_rank = grp.shape[0]
    return grp.sort_values(col, ascending=False).head(return_rank)[['State']]


def hypothesis3(testingData3: pd.DataFrame):
    """
    This function consists of steps required to perform hypothesis 3.
    :param testingData3: Dataframe consisting of statewise vaccine and test data.
    """
    print("#########################################################################################################")
    print("                                            HYPOTHESIS 3                                                ")
    print("#########################################################################################################")

    plot = go.Figure(
        data=[go.Bar(name='Covaxin', x=testingData3['State'], y=testingData3['Total Covaxin Administered']),
              go.Bar(name='CoviShield', x=testingData3['State'], y=testingData3['Total CoviShield Administered'])])
    plot.update_layout(
        title={
            'text': 'Number of Covaxin And CoviShield Administered per State',
            'y': 0.9,
            'x': 0.4,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="State",
        yaxis_title="Count of vaccine administered",
        # xaxis = dict(
        # tickfont=dict(color="#ff7f0e"),
        # tickmode = 'array',
        # tickvals = [19, 31],
        # ticktext = ['Maharashtra', 'Telangana']
        # )

    )
    plot.show()

    statewise_vaccination = pd.read_csv('http://api.covid19india.org/csv/latest/cowin_vaccine_data_statewise.csv',
                                        parse_dates=['Updated On'])

    idx = statewise_vaccination.State == 'India'

    india = statewise_vaccination.loc[idx, :]
    statewise_vaccination = statewise_vaccination.loc[~idx, :]

    covis_top_3 = statewise_vaccination.groupby(['Updated On']).apply(rank_, 'Total CoviShield Administered', 3) \
        .reset_index().drop('level_1', axis=1)
    covis_top = covis_top_3.drop_duplicates('Updated On', keep='first')

    covx_top_3 = statewise_vaccination.groupby(['Updated On']).apply(rank_, 'Total Covaxin Administered', 3) \
        .reset_index().drop('level_1', axis=1)
    covx_top = covx_top_3.drop_duplicates('Updated On', keep='first')

    title_list = ['No. of Days when state received most daily shares of Coviesheild (State in top 3)',\
                  'No. of Days when state received most daily shares of Coviesheild (State tops the list)',\
                  'No. of Days when state received most daily shares of Covaxin (State in top 3)',\
                  'No. of Days when state received most daily shares of Covaxin (State tops the list)']
    i = 0

    for df in [covis_top_3, covis_top, covx_top_3, covx_top]:
        df = df.State.value_counts().reset_index().rename(columns={'index': 'State', 'State': 'Highest reciever in'})
        fig = go.Figure(go.Pie(labels=df.State, values=df['Highest reciever in']))
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            title={
                'text': title_list[i],
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )
        i = i + 1
        fig.show()


if __name__ == '__main__':

    #Loading all the input files for Hypothesis 1
    stateElections = pd.read_csv("./StateElectionData.csv")
    stateVaccinations = pd.read_csv("./covid_vaccine_statewise.csv")
    statePopulation = pd.read_csv("./statePopulationIndia.csv")
    stateVaccinations = cleanData(stateVaccinations)
    stateVaccineRecords = getStateVaccineRecords(stateVaccinations)

    # Cleaning all the file for Hypothesis 1
    stateElections = cleanData(stateElections)

    statePopulation = cleanData(statePopulation)

    # Performing analysis for Hypothesis 1
    stateGovt = hypothesis1(stateElections, stateVaccinations, statePopulation)

    # Loading all the input files for Hypothesis 2
    stateTesting = pd.read_csv("./statewise_tested_numbers_data.csv")

    # CLeaning all the files for Hypothesis 2
    stateTesting = cleanData(stateTesting)

    # Performing analysis for Hypothesis 2
    hypothesis2(stateTesting, stateVaccinations, stateGovt)


    # Grouping all the data for Hypothesis 3
    groupedVaccineTypeData = getVaccineTypeCount(stateVaccinations)

    # Performing analysis for Hypothesis 3
    hypothesis3(groupedVaccineTypeData)





