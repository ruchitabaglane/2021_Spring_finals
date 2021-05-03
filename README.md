# 2021_Spring_finals

# Influence of Indian Politics on COVID-19 vaccination

## The idea behind the project

The number of new Covid-19 cases are increasing day by day around the world.In some countries the number is increasing even after vaccines being made available. One of the reasons for this may be the distribution of vaccine in that particular country. Many factors affect the distribution of vaccine such as political, economic, demographic, etc. Additionally, as per some articles the distribution of vaccine can be affected by the political lean of the country. The sudden surge in number of Covid-19 cases in India even after vaccinations being made available intrigued us in exploring this topic at depth.  
## Project Description

Our project aims to analyze various factors affecting provision of Covid-19 vaccination in India. We intend to find the correlation between political factors and vaccination distribution in each state. Further, we also aim to find how does the availability of vaccination affects the number of positive cases for each state in India.

## Team Members

1. Ruchita Baglane (Github id - github.com/ruchitabaglane)
2. Prachi Doshi (Github id - github.com/PrachiDoshi22)

## Hypothesis 1

Conducting a test to measure the vaccine distribution for states having common government at state and center level VS states having different government at state and center level.

H0 : Distribution of vaccination is better in states having common government at state and center level. <br>
HA : No effect of governing body/party on the distribution of vaccination.

We first classified the states based on the political party currently in power. Then we calculated the rate of vaccinations by dividing total number of individuals vaccinated  by total population of states falling under each political party.

Below is the plot for each statewise ruling party vs vaccination rate.
![image](https://user-images.githubusercontent.com/77983776/116843276-15e1b880-aba5-11eb-9e19-1ed7be9e963b.png)

By looking at the first plot, we can say that the states having BJP in power which also the current ruling party in India, have lower vaccination rates. Whereas the states with ruling parties like NCP and SKM have better vaccination rate. Thus, it is not the case that the states with same ruling party as that of the centre have a better vaccination rate. 

For better understanding, we also plotted a bar graph of statewise vaccination rates. Each state is coloured to represent the currently ruling party in that state. It can be observed that even though BJP is in power in most of the states, it has a significantly low vaccination rates in states like Uttar Pradesh, Madhya Pradesh, Bihar, Jharkhand, Assam. Whereas, the parties like SKM and NCP have better vaccination rates in states they are in power.
![image](https://user-images.githubusercontent.com/77983776/116843786-a1a81480-aba6-11eb-8260-b35cc8a8a1ae.png)

Thus, we can accept the alternate hypothesis that the governing body/party has no effect on the distribution of vaccination.


## Hypothesis 2

Conducting a test to see if the availability of vaccine has had any impact on number of Covid-19 tests being performed.

H0 : Administration of vaccine has impacted the number of Covid-19 tests being conducted(increased or decreased).<br>
HA : Vaccine had no impact on the number of Covid-19 tests being conducted.

Based on the results of the above hypothesis we may further check if the actual number of positive cases increase or decrease after vaccination (Since in India the number of cases has increased even after vaccine distribution)

## Hypothesis 3

Does political relations of state producing the vaccine has an impact on the accessibility of vaccine to other states.

H0 : States having good political relations with states producing the vaccine has better accessibility to the vaccine.<br>
HA : Accessibility to vaccine in any state is not dependent on the political relations with the state producing the vaccine.

## Datasets Used

India Elections:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/26526
https://www.kaggle.com/awadhi123/indian-election-dataset

Covid-19 Vaccination: 

https://www.kaggle.com/gpreda/covid-world-vaccination-progress
https://www.kaggle.com/sudalairajkumar/covid19-in-india?select=covid_vaccine_statewise



