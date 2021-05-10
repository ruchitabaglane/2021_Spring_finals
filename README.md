# 2021_Spring_finals

# Influence of Indian Politics on COVID-19 vaccination

## The idea behind the project
Ideally, the number of new Covid-19 positive cases should not increase once vaccinations are availabe, right? But that's not the case with majority nations around the globe. We thought of probing the matter for a particular nation, India - the current worst COVID hit nation. As we understand, the distribution and availability of vaccinations can be attributed to various factors like politics, economics, demographics, weather, etc. Intrigued! we started out on a voyage to find what affects India particularly. 

## Project Description
Our project aims to analyze various factors affecting provision of Covid-19 vaccination in India. We intend to find the correlation between political lean and vaccine distribution across states in India. Further, we aim to find how vaccination drives affect the number of tests conducted and number of people testing positive from a nation’s perspective. Finally, we seek to explore if a state with manufacturing unit of the vaccine gets a preferential vaccine distribution.

## Team Members

1. Ruchita Baglane (Github id - github.com/ruchitabaglane)
2. Prachi Doshi (Github id - github.com/PrachiDoshi22)

## Hypothesis 1

Does having the same ruling party at state and nation level (i.e. positive political lean) influence the vaccine distributed to that state?

H0 : Distribution of vaccination is better in states having common ruling party at state and center level. <br>
HA : No effect of ruling party on the distribution of vaccine.

We first classified the states based on the political party currently in power. Then compared the total number of individuals vaccinated under each political party.

## Below is the plot for each state wise ruling party vs total individuals vaccinated.

![image](https://user-images.githubusercontent.com/77983551/117612984-c0208980-b12b-11eb-820a-328c53c7690c.png)

By looking at the above graph one could be misled to believe that BJP-the national ruling party has an unprecedented bias in the availability of vaccines with 100M people vaccinated and the best standing at 10M people. 
But political parties govern a varied portion of the nation's population. Which lead us to compute and compare the vaccination rate instead. 

Vaccination rate = (Total number of people vaccinated/Population) * 100

## Below is the plot for each state wise ruling party vs vaccination rate.
![image](https://user-images.githubusercontent.com/77983551/117613101-f4944580-b12b-11eb-8f80-70c4a68aa56c.png)

By looking at the above plot, we can say that the states having BJP-the national ruling party in power, have lower vaccination rates compared to other parties. Thus, it is not the case that the states with positive political lean have a better vaccination rate. 

For better understanding, we have also plotted a bar graph of state wise vaccination rates. States with common ruling party are grouped and marked with the same color. It can be observed that even though BJP is in power in most of the states, it has a significantly low vaccination rates in states like Uttar Pradesh, Madhya Pradesh, Bihar, Jharkhand, Assam. Whereas the parties like SKM and NCP have better vaccination rates in states where they are in power.

![image](https://user-images.githubusercontent.com/77983776/117625672-de8e8100-b13b-11eb-9641-0c4a5e2a9f27.png)

Thus, <em> we can accept the alternate hypothesis </em> that the ruling party has no visible effect on the distribution of vaccines, and we need to explore more factors which could have an impact on the vaccine distribution across the states.

## Hypothesis 2

Does availability of vaccine has an influence on number of Covid-19 tests being performed?

H0 : Administration of vaccine has impacted the number of Covid-19 tests being conducted(positively or negatively).<br>
HA : Vaccine had no impact on the number of Covid-19 tests being conducted.

![image](https://user-images.githubusercontent.com/77983776/117695353-c2173680-b185-11eb-802c-6c1ae223a5cc.png)

As we can see the number of tests conducted reduced after administration of first dose of the vaccine. There is also a subsequent rise in the number of people tested and the positive cases with the rise in administration of the second dose of the vaccine.

Thus, <em> we accept the null hypothesis </em> that the administration of vaccine has an influence on the number of Covid-19 tests being conducted(increased or decreased).

## Hypothesis 3

Does the political relations of states producing the vaccine have an impact on the accessibility of the vaccine to the other states.

H0 : States having good political relations with states producing the vaccine has better accessibility to the vaccine.<br>
HA : Accessibility to vaccine in any state is not dependent on the political relations with the state producing the vaccine.

## The plot below shows the number of Covaxin and CoviShield distributed per state

![image](https://user-images.githubusercontent.com/77983551/117613637-cf540700-b12c-11eb-873b-c342d2d41e88.png)

## The plots belows represents the top consumers(states) of Covaxin and CoviShield

![image](https://user-images.githubusercontent.com/77983776/117695472-e1ae5f00-b185-11eb-8f83-8389ad0e1ab0.png)
The above chart represents the number of days for which each state was in the top 3 consumers of CoviShield. <br> <br>


![image](https://user-images.githubusercontent.com/77983776/117695518-ee32b780-b185-11eb-87c4-ae1c74991087.png)
The above chart represents the number of days for which each state was the top consumer of CoviShield. <br>
Maharashtra topped the list for 40% of days.<br>
Rajasthan topped the list for 16.5% of days.<br>
Uttar Pradesh topped the list for 33.9% of days.<br><br>


![image](https://user-images.githubusercontent.com/77983776/117695570-fd196a00-b185-11eb-9bca-3ed2d503056d.png)
The above chart represents the number of days for which each state was in the top 3 consumers of Covaxin.<br><br>


![image](https://user-images.githubusercontent.com/77983776/117695613-073b6880-b186-11eb-8a2e-1c0712d844e3.png)
The above chart represents the number of days for which each state was the top consumer of Covaxin. <br>
Maharashtra topped the list for 40.9% of days.<br>
Gujarat topped the list for 45.2% of days.<br>
Odisha topped the list for 9.57% of days.<br><br>

Covaxin produced in Telangana (Ruling Party - Bharatiya Janata Party)<br>
CoviShield produced in Maharashtra (Ruling Party - Bharatiya Janata Party)<br>

From the above analysis and from the plot 3 of Hypothesis 1, we can see that the distribution of Covaxin in a particular state is independent of the political relations of the states with the state producing this vaccine variant. As even the states with ruling party apart from Bharatiya Janata Party have topped the list for several number of days in the distribution of both Covaxin (E.g. Odisha, Andhra Pradesh, West Bengal). Moreover, Telangana the producing state of Covaxin never has the maximum share of daily vaccines.<br> On the other hand, Covishield is manufactured by Maharashtra and it is also the top consumer of the vaccine, thus indicating a producer bias. Further Gujrat and Rajasthan the next 2 top consumers of Covishield are also ruled by Bharatiya Janata Party. Thus, indicating a political relations bias as well. Hence, <em> we reject the alternate hypothesis </em>. 


## Datasets Used

India Elections:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/26526 <br>
https://www.kaggle.com/awadhi123/indian-election-dataset<br>

Covid-19 Vaccination: 

https://www.kaggle.com/gpreda/covid-world-vaccination-progress <br>
https://www.kaggle.com/sudalairajkumar/covid19-in-india?select=covid_vaccine_statewise <br>
https://api.covid19india.org/csv/latest/statewise_tested_numbers_data.csv <br>
https://api.covid19india.org/csv/latest/case_time_series.csv <br>




