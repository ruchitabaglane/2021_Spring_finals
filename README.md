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

## Below is the plot for each statewise ruling party vs total individuals vaccinated.

![image](https://user-images.githubusercontent.com/77983551/117612984-c0208980-b12b-11eb-820a-328c53c7690c.png)

## Below is the plot for each statewise ruling party vs vaccination rate.

![image](https://user-images.githubusercontent.com/77983551/117613101-f4944580-b12b-11eb-8f80-70c4a68aa56c.png)

By looking at the above plot, we can say that the states having BJP in power which is also the current ruling party in India, have lower vaccination rates. Thus, it is not the case that the states with same ruling party at the state and centre level have a better vaccination rate. 

For better understanding, we have also plotted a bar graph of statewise vaccination rates. The colour of the bar represents the currently ruling party in that state. It can be observed that even though BJP is in power in most of the states, it has a significantly low vaccination rates in states like Uttar Pradesh, Madhya Pradesh, Bihar, Jharkhand, Assam. Whereas, the parties like SKM and NCP have better vaccination rates in states that they are in power.

![image](https://user-images.githubusercontent.com/77983551/117613288-476dfd00-b12c-11eb-91e2-3c1018dfcd16.png)

Thus, <em> we can accept the alternate hypothesis </em> that the governing body/party has no effect on the distribution of vaccination and we need to explore more factors which has an impact on the vaccine distribution across the states.


## Hypothesis 2

Conducting a test to see if the availability of vaccine has any influence on number of Covid-19 tests being performed.

H0 : Administration of vaccine has impacted the number of Covid-19 tests being conducted(increased or decreased).<br>
HA : Vaccine had no impact on the number of Covid-19 tests being conducted.


So as we can see from the graph that the number of tests conducted has increased after the admistration second fose of the vaccine. Also, we can see a spike in the number of positive cases shortly after the second dose of vaccine started.

Thus, <em> we accept the null hypothesis </em> that the admistration of vaccine has impacted the number of Covid-19 tests being conducted(increased or decreased).

## Hypothesis 3

Does the political relations of states producing the vaccine have an impact on the accessibility of the vaccine to the other states.

H0 : States having good political relations with states producing the vaccine has better accessibility to the vaccine.<br>
HA : Accessibility to vaccine in any state is not dependent on the political relations with the state producing the vaccine.

## The plot below shows the number of Covaxin and CoviShield distributed per state

![image](https://user-images.githubusercontent.com/77983551/117613637-cf540700-b12c-11eb-873b-c342d2d41e88.png)

## The plots belows represents the top consumers(states) of Covaxin and CoviShield

![image](https://user-images.githubusercontent.com/77983551/117613761-fdd1e200-b12c-11eb-9b92-4628be057ce7.png)
The above chart represents the number of days for which each state was in the top 3 consumers of CoviShield. <br> <br>


![image](https://user-images.githubusercontent.com/77983551/117613834-180bc000-b12d-11eb-83a3-f2e8ded24d19.png)
The above chart represents the number of days for which each state was the top consumer of CoviShield. <br>
Maharashtra topped the list for 40% of days.<br>
Rajasthan topped the list for 16.5% of days.<br>
Uttar Pradesh topped the list for 33.9% of days.<br><br>


![image](https://user-images.githubusercontent.com/77983551/117613904-2e198080-b12d-11eb-893b-513e098809f3.png)
The above chart represents the number of days for which each state was in the top 3 consumers of Covaxin.<br><br>


![image](https://user-images.githubusercontent.com/77983551/117613985-48535e80-b12d-11eb-93a7-7441b990d3a7.png)
The above chart represents the number of days for which each state was the top consumer of Covaxin. <br>
Maharashtra topped the list for 40.9% of days.<br>
Gujarat topped the list for 45.2% of days.<br>
Odisha topped the list for 9.57% of days.<br><br>

Covaxin produced in Telangana (Ruling Party - Bharatiya Janata Party)<br>
CoviShield produced in Maharashtra (Ruling Party - Bharatiya Janata Party)<br>

Thus from the above analysis and from the plot 3 of Hypothesis 1, we can see that the distribution of Covaxin and CoviShield in a particular state is independent of the political relations of the states with the state producing these vaccines. The reason for this is that even the states with ruling party apart from Bharatiya Janata party have topped the list for sevral number of days in the distribution of both CoviShield and Covaxin (Eg. Odisha, Andhra Pradesh, West Bengal).<br>
Hence, <em> we accept the alteranate hypothesis </em>.


## Datasets Used

India Elections:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/26526 <br>
https://www.kaggle.com/awadhi123/indian-election-dataset<br>

Covid-19 Vaccination: 

https://www.kaggle.com/gpreda/covid-world-vaccination-progress <br>
https://www.kaggle.com/sudalairajkumar/covid19-in-india?select=covid_vaccine_statewise <br>
https://api.covid19india.org/csv/latest/statewise_tested_numbers_data.csv <br>
https://api.covid19india.org/csv/latest/case_time_series.csv <br>




