# When Psychology Meets Product Analytics: A Rule-Based Behavioural Framework for Predicting SaaS Trial Conversion.


## Table of Contents

1. [Project Overview](#1-project-overview)

2. [Business Context & Psychological Framework](#2-business-context--psychological-framework)

3. [Dataset Description](#3-dataset-description)

4. [Tools & Technologies](#4-tools--technologies)

5. [Data Generation Methodology](#5-data-generation-methodology)

6. [Data Cleaning & Validation](#6-data-cleaning--validation)

7. [Exploratory Data Analysis](#7-exploratory-data-analysis)

   7.1 Customer Demographics Overview

   7.2 Trial Behaviour Overview

   7.3 Conversion Rate Overview

   7.4 Behavioural Friction Indicators

8. [Feature Engineering](#8-feature-engineering)

9. [Behavioural Friction Analysis](#9-behavioural-friction-analysis)

    9.1 Cognitive Load Analysis

    9.2 Present Bias Analysis

    9.3 Early Activation Analysis

    9.4 Friction Point Identification

10. [Conversion Risk Scoring (Rule based scoring using behavioural thresholds)](#10-conversion-risk-scoring)

11. [Key Findings & Business Recommendations](#11-key-findings--business-recommendations)

12. [Limitations](#12-limitations)

13. [Conclusion](#13-conclusion)



## Project Overview

This project bridges behavioural psychology and product analytics to understand why SaaS trial users do not convert to paid customers. Using Python for data generation, feature engineering and machine learning, Azure SQL for cloud data storage, and Looker Studio for visualisation, this analysis identifies psychological friction points in the trial journey and builds a conversion risk scoring model to help SaaS businesses intervene before users churn in order to help teams have a data-driven idea of the amount of resources needed to gain paid clients.

<img width="623" height="389" alt="Screenshot 2026-06-19 232842" src="https://github.com/user-attachments/assets/58b64780-69fe-44bc-95ba-166b09b8364b" />

Link to report: https://datastudio.google.com/s/sNePQGyN2p4



## Business Context & Psychological Framework


### Business Context

NovaMind is a fictional B2B SaaS company offering a project management and collaboration tool. The company provides a 14-day free trial with full feature access. After the trial, users either convert to a paid plan or churn. Despite strong trial sign-up numbers, conversion rates remain below industry expectations, and the business lacks visibility into which users are at risk and why. The current available plans for the business are: basic plan that starts at $29/month, pro at $79/month and enterprise at $199/month.


### Psychological Framework

This analysis applies established behavioural psychology concepts to product analytics. The first is cognitive load theory (Sweller, 1988). When users encounter too many features or a complex interface, mental effort increases and engagement drops. In SaaS products, cognitive overload during onboarding is one of the leading causes of trial abandonment. This is measured in this analysis through the complexity score, a derived metric capturing the ratio of features explored to time invested. 

Second is, present bias (Kahneman & Tversky). Users consistently overvalue immediate comfort over future benefit. Trial users intend to convert "later" but never do. This manifests as last minute activity where users who only engage heavily in the final days of their trial, suggesting they procrastinated on evaluating the product until it was too late.

Next is, loss aversion (Kahneman & Tversky).This is when users feel the pain of losing access to trial features more strongly than the pleasure of gaining paid features. Understanding which features users engage with most helps identify what they stand to lose and how to frame conversion messaging around that loss.

Then there is the early activation principle. Research from Intercom, HubSpot and Mixpanel consistently shows that users who engage meaningfully within the first 3 days of a trial convert at significantly higher rates. Early activation signals genuine intent and product-market fit at the individual user level.
 
Lastly, Fogg behaviour model (BJ Fogg). Behaviour occurs when motivation, ability and a trigger align simultaneously. Applied to SaaS conversion, a user must want to convert (motivation), find it easy to convert (ability), and receive a timely nudge (trigger). Friction analysis identifies where this alignment breaks down.



## Dataset Description

This dataset was synthetically generated using Python based on established SaaS conversion benchmarks and behavioural psychology research. All user names, emails, and company details are fictitious. Dataset includes 5000 users with a 15 day trial spanning across 2023 - 2024. The data generation methodology is documented in Section 5


## Tools & Technologies

Tool
Purpose
Python
 Data generation, cleaning, EDA, ML modelling
Pandas
Data manipulation and analysis
NumPy
Numerical computations
Fake
Synthetic data generation
Supabase
Cloud data storage
Supabase Postgres Database
Database management
Data Studio (Looker Studio)
Dashboard and visualisation
GitHub
Version control and documentation



## Data Generation Methodology

The dataset was synthetically generated using python's numpy, pandas and faker libraries. Rather than using a random dataset, behavioural patterns were deliberately embedded based on established SaaS industry benchmarks and behavioural psychology research. Conversion at was set at 22% reflecting the industry average for B2B SaaS trial conversion reported by Intercom and HubSpot. Plan choice was weighted by company size to reflect real B2B purchasing behaviour, that is,larger companies were more likely to choose Enterprise plans while smaller companies favoured Basic plans. 

To assess user pattern, a higher login frequency was set at (15-50 logins). Longer session duration was set to (20-90 minutes), more features explored (5-10 out of 10),  higher onboarding completion rate (85%),stronger early activation (80% logged in within the first 3 days). To assess the churn user pattern, lower login frequency was set to (1-14 logins), shorter session durations (2-25 minutes). fewer features explored (1-4 out of 10), lower onboarding completion rate (25%), weaker early activation (30% logged in within first 3 days) and a higher support ticket rates indicating friction. The complexity score was derived as: features_used /  avg_session_duration / by 10. A high score indicates a user explored many features in a short time, a proxy for cognitive overload consistent with Sweller's Cognitive Load Theory.

A fixed random seed of 42 was used across all libraries to ensure full reproducibility of the dataset. The generated dataset was pushed to a Supabase PostgreSQL cloud database using SQLAlchemy and the pandas to_sql() function. All subsequent analysis pulls data directly from the cloud database using pd.read_sql(), mimicking a real world production data pipeline where analysts query live databases rather than working with local files. 


## Data Cleaning & Validation
The initial check confirms that the dataset is perfectly clean and complete, showing a total of 5,000 trial users, 25 columns, and zero duplicate accounts. All numbers fall strictly within realistic boundaries for NovaMind’s 14-day trial timeline. For example, active days stop exactly at 14, and the time spent per session maxes out at 90 minutes. This proves there are no impossible dates or corrupt rows in the database.
Trial users are evenly spread across different countries, industries, and company sizes, with roughly 500 users per country or industry. This balanced mix is excellent because it prevents our future predictive models from favoring one specific group over another. Out of the users who converted, the Pro plan ($79/month) is the most popular choice with 387 buyers, followed closely by the Enterprise and Basic tiers. Because plan choices are tied to company size, larger corporations drive the bulk of NovaMind's trial revenue. This insight shows the business exactly where to focus its human sales resources.
The data confirms that exactly 1,119 users bought a subscription, giving NovaMind an overall conversion rate of 22.38%, which perfectly matches real-world averages for successful business software. The remaining 3,881 users chose to walk away, giving us an uncorrupted baseline to study why people lose interest. Crucially, the timeline tracking conversion speed contains zero missing values for paying customers and is strictly empty for the 3,881 churned users. This perfect tracking confirms the data pipeline is functioning correctly, allowing a safe predictive model without risking errors or information leaks.


## Exploratory Data Analysis

  ### Customer Demographics Overview

The customer base is evenly distributed across countries, industries, and company size segments, indicating that the synthetic dataset is not dominated by any single demographic group. This balanced representation reduces the risk of demographic bias in downstream analysis and allows behavioural factors such as activation, engagement, onboarding completion, and feature adoption to play a larger role in explaining trial conversion outcomes. 


   ### Trial Behaviour Overview

This analysis examines how users interacted with the product during their free trial period. The goal is to understand whether users actively explored the platform or showed signs of limited engagement before deciding whether to convert to a paid plan.
The results show that user behaviour is not evenly distributed. Most users fall into one of two broad groups: highly engaged users who logged in frequently, spent more time using the product, and explored more features, and less engaged users who interacted with the platform only occasionally. This suggests that trial users experience the product very differently, which may influence their likelihood of becoming paying customers.
The login and session duration distributions indicate that many users also either invested meaningful time in evaluating the product or only spent a short period exploring it before disengaging. A similar pattern appears in feature usage, where some users explored a wide range of functionality while others interacted with only a small number of features.
The day's active distribution shows that many users were only active during part of their trial period. This may indicate that some users lost interest before experiencing the full value of the product. In SaaS businesses, sustained engagement throughout the trial period is often a positive sign of purchase intent.
The complexity score distribution is heavily concentrated among lower values, suggesting that most users experienced relatively low levels of feature exploration relative to time spent in the product. A smaller group of users displayed higher complexity scores, indicating more intensive exploration behaviour. Support ticket activity was spread across multiple levels, showing that users experienced varying degrees of assistance needs during the trial. 


  ### Conversion Rate Overview

The overall trial-to-paid conversion rate was 22.4%, meaning that approximately one in every five trial users became a paying customer while the remaining users churned before purchasing a subscription. This conversion rate is broadly consistent with commonly reported benchmarks for B2B SaaS products.
Conversion rates varied across company sizes. Mid-sized organisations with 51–200 employees recorded the highest conversion rates, while companies with 201–500 employees showed the lowest. This suggests that company size may influence purchasing behaviour, although the differences are relatively modest.
Industry-level analysis also revealed variation in conversion performance. Technology, Legal, and Healthcare organisations achieved the highest conversion rates, while Real Estate, Finance, and Marketing organisations converted at lower rates. These differences may reflect varying levels of software adoption, operational requirements, and perceived product value across industries.
Overall, demographic characteristics appear to have some influence on conversion outcomes. However, these differences alone are unlikely to fully explain why users convert. Further analysis of behavioural metrics such as engagement levels, onboarding completion, feature adoption, and product usage patterns is required to identify the strongest drivers of conversion.


   ## Behavioural Friction Indicators

This analysis compares the behaviour of users who converted to a paid subscription against those who churned during the trial period. The objective is to identify behavioural patterns that may indicate friction or barriers preventing successful conversion.
The results show clear differences between the two groups. Users who converted logged in more frequently, remained active for more days, explored a greater number of product features, and spent significantly longer using the platform compared to users who churned. This suggests that meaningful engagement during the trial period plays an important role in conversion success.
The complexity score analysis revealed that churned users generally recorded higher complexity scores than converted users. The higher complexity scores observed among churned users suggest that increased product complexity or unstructured feature exploration may create friction during the trial experience. This finding is consistent with Cognitive Load Theory, which proposes that excessive mental effort can reduce engagement and hinder decision-making.
Early activation emerged as one of the strongest behavioural indicators of conversion. Users who logged into the platform within the first three days of their trial converted at a substantially higher rate than users who delayed engagement. This supports the Early Activation Principle, which suggests that experiencing value early in the customer journey is an important driver of long-term adoption and conversion. 
Users who completed the onboarding process converted at significantly higher rates than those who did not. The strong relationship between onboarding completion and conversion aligns with the Fogg Behaviour Model. Completing onboarding may increase user ability by reducing uncertainty and improving understanding of the product, making conversion behaviour more likely when purchase opportunities are presented. 
Overall, the findings indicate that behavioural factors are stronger indicators of conversion than demographic characteristics. User engagement, early activation, and onboarding completion appear to be key drivers of successful trial-to-paid conversion, while higher complexity scores may signal potential friction points within the user journey.


  ## Feature Engineering
To better capture user behaviour during the trial period, several new features were created from the original dataset. These engineered features combine multiple user actions into broader behavioural indicators that can be used for analysis and predictive modelling.
The engagement score measures overall platform activity by combining login frequency, days active, and average session duration. Higher scores indicate users who interacted more frequently and consistently with the product.
The product adoption score measures how deeply users explored and utilised the platform. It combines features used, reports generated, and integrations enabled to reflect overall product adoption.
The collaboration score measures team engagement by evaluating the number of team members invited to the platform. Higher scores suggest greater collaborative usage.
The friction score was designed to identify potential obstacles experienced during the trial journey. It combines product complexity, support ticket activity, and inactivity patterns, with higher scores indicating greater friction.
The recency score measures how recently a user interacted with the platform. Users who remained active closer to the end of the trial receive higher scores, while less recently active users receive lower scores.
Engineered features were created for behavioural analysis purposes. During initial modelling attempts, these features were found to cause data leakage due to the synthetic nature of the dataset. The project was subsequently reoriented toward a rule-based behavioural scoring approach, which better aligns with the analytical rather than predictive nature of this work. 


  ## Behavioural Friction Analysis

  <img width="625" height="389" alt="Screenshot 2026-06-19 233235" src="https://github.com/user-attachments/assets/859575ca-7dfc-49c3-b8f5-cfaa05af8032" />

 Link to report: https://datastudio.google.com/s/ov__D2y4g9Q



  ### Cognitive Load Analysis

Cognitive Load Theory (Sweller, 1988) suggests that when people experience excessive mental effort while learning a product, engagement decreases and abandonment becomes more likely. To assess this, complexity score, features used and average session duration were analysed across converted and churned users.
Converted users demonstrated substantially lower complexity scores (1.78) compared to churned users (2.80), while also exploring more features (6.57 vs 3.52) and spending longer in the platform (50.68 minutes vs 21.35 minutes). This implies that successful users were able to navigate the product more efficiently and derive value without experiencing excessive cognitive strain.
Further analysis showed that users classified within the Low Cognitive Load segment achieved a conversion rate of 23.1%, while conversion dropped to 0.6% for Medium Load users and 0% for High Load users. The high cognitive load tier contained only 39 users (0.78% of the dataset), which is insufficient for statistically reliable conclusions. Due to the 0% conversion rate observed in this segment which was a reflection of the synthetic data generation methodology rather than a genuine business insight. Analysis of cognitive load impact is therefore based primarily on the Low and Medium load segments which contain adequate sample sizes. This indicates an increasing cognitive complexity is strongly associated with lower conversion outcomes. These findings support cognitive load theory and suggest that simplifying onboarding experiences, reducing unnecessary complexity and guiding users towards key features may improve trial-to-paid conversion rates. 


  ### Present Bias Analysis

Present Bias, described by Kahneman and Tversky, suggests that individuals tend to prioritise immediate comfort over future benefits. Within SaaS trials, this often manifests as users postponing product exploration until the final stages of the trial period.
Users exhibiting last-minute activity achieved a conversion rate of 15.5%, compared to 26.9% among users who engaged consistently throughout the trial. This indicates that users who delayed evaluation were significantly less likely to convert.
Engagement timing also differed considerably between converted and churned users. Converted users remained engaged until approximately Day 10.6 of the trial, whereas churned users disengaged around Day 5.0. This suggests that sustained engagement throughout the trial is associated with higher purchase intent.
Recency analysis further reinforced this pattern. Users who logged in within the previous two days converted at a rate of 58.3%, compared to 32.2% among moderately active users and only 8.6% among inactive users.
These findings support Present Bias theory by demonstrating that delayed engagement and prolonged inactivity significantly reduce conversion likelihood. The results suggest that timely reminders, proactive engagement campaigns and early product education may help reduce procrastination behaviours during trials.


  ### Early Activation Analysis

The Early Activation Principle is one of the most consistently validated findings within SaaS product research. Studies conducted by Intercom, HubSpot and Mixpanel show that users who achieve meaningful engagement during the first few days of a trial are substantially more likely to become paying customers.
Users who logged into the platform within the first three days converted at a rate of 32.9%, compared to only 12.7% among users who failed to activate early. This represents a conversion lift of approximately 2.6 times.
Early activated users also demonstrated stronger behavioural engagement across multiple dimensions, including feature exploration, login frequency and active usage days. These behaviours indicate stronger product adoption and a higher likelihood of discovering product value during the trial period.
The combined effect of early activation and onboarding completion was particularly significant. Users who neither activated early nor completed onboarding converted at only 5.6%, while users who completed both activities achieved a conversion rate of 51.9%.These findings strongly reinforce established SaaS industry research and suggest that the first few days of the trial period represent the most critical stage of the customer journey.


  ### Friction Point Identification

The Fogg Behaviour Model proposes that behaviour occurs when motivation, ability and triggers are present simultaneously. Conversion friction occurs when one or more of these factors are weakened.
Onboarding Completion: Users who completed onboarding converted at a rate of 38.1%, compared to only 9.7% among users who did not complete onboarding. This suggests that onboarding significantly improves a user's ability to understand and adopt the platform.

Support Ticket Friction: support ticket activity showed a clear negative relationship with conversion. Users raising no support tickets converted at 27.6%, compared to 23.3% for users with one to two tickets and 19.6% for users with three or more tickets. This indicates that unresolved friction and usability issues reduce purchase likelihood.
Feature Adoption: feature adoption emerged as one of the strongest behavioural signals. Users with low feature adoption converted at 6.7%, compared to 16.7% for medium adopters and near-complete conversion among high adopters. This suggests that deeper product exploration significantly increases perceived value.
Collaboration Signals : collaboration behaviour also demonstrated a strong relationship with conversion. Users who invited three or more team members achieved a conversion rate of 43.5%, compared to approximately 10% among users with little or no collaboration activity.
This finding aligns with SaaS industry research showing that collaborative workflows increase product stickiness, strengthen perceived value and create higher switching costs. Collectively, these findings indicate that onboarding completion, feature adoption, collaboration and friction reduction represent key opportunities for improving trial conversion performance.


 ### Conversion Risk Scoring (Rule based scoring using behavioural thresholds)

<img width="615" height="384" alt="Screenshot 2026-06-20 000122" src="https://github.com/user-attachments/assets/f6ba100f-02a6-4d23-9d18-8f64b548b0fe" />

Link to report: https://datastudio.google.com/s/vgL89DWa9Ec

Rather than relying on predictive machine learning models, a rule-based behavioural scoring framework was developed to assess conversion risk. This approach was selected because it provides complete transparency and allows every score to be directly linked to observable user behaviour and established behavioural psychology principles.
The scoring system evaluates users across six primary behavioural dimensions: early activation, onboarding completion, engagement depth, feature adoption, collaboration activity and recency of engagement
Behavioural friction indicators such as support ticket volume, cognitive load and last-minute activity were incorporated as penalty factors.
Users were segmented into four risk categories:

High Risk (0–25)
Medium Risk (26–50)
Low Risk (51–75)
Very Likely to Convert (76–100)
The average behavioural risk score across all users was 36.0 out of 100.
Risk segmentation produced meaningful separation between user groups:

Segment
Users
Actual Conversion Rate
High Risk
1,907
1.6%
Medium Risk
1,764
13.0%
Low Risk
975
52.7%
Very Likely to Convert
354
95.2%


The substantial differences between segments indicate that behavioural signals successfully distinguish users with varying levels of purchase intent. This framework provides a practical method for customer success and growth teams to prioritise interventions and allocate resources towards users who exhibit high churn risk.


## Key Findings & Business Recommendations

 ### Key Findings

Several consistent behavioural patterns emerged throughout the analysis.
Behavioural engagement is the strongest conversion driver.
Demographic factors such as industry, company size and country showed only modest differences in conversion rates. Whereas, behavioural metrics demonstrated substantial separation between converted and churned users. Users who logged in frequently, remained active longer, explored more features and spent more time using the platform converted at significantly higher rates.

Early activation is critical.
Users who logged into the platform within the first three days converted at a rate of 32.9%, compared to only 12.7% among users who did not activate early. This represents a 2.6x increase in conversion likelihood and strongly supports the early activation principle commonly reported in SaaS research by agencies such as: Intercom, HubSpot and Mixpanel.
Onboarding completion has a major impact
Users who completed onboarding converted at 38.1%, while users with incomplete onboarding converted at only 9.7%. This finding aligns with the Fogg Behaviour Model, suggesting that effective onboarding increases user ability and reduces friction during product adoption.

Cognitive load reduces conversion
Churned users experienced higher complexity scores and lower feature adoption than converted users. Conversion rates dropped sharply as cognitive load increased, supporting Cognitive Load Theory (Sweller, 1988), which proposes that excessive mental effort negatively affects engagement and decision-making.

Present bias affects trial behaviour
Users who exhibited last-minute activity patterns converted less frequently than users who engaged consistently throughout the trial. Additionally, converted users remained engaged until approximately Day 10.6 of the trial, while churned users disengaged around Day 5.0. This supports the concept of present bias, where users postpone evaluation until it becomes too late to experience meaningful product value.

Collaboration and product adoption increase conversion
Users who invited three or more team members converted at 43.5%, compared to roughly 10% among users with little or no collaboration activity. Similarly, deeper feature adoption was associated with dramatically higher conversion rates. These findings reinforce SaaS industry research showing that collaborative usage and product integration increase product stickiness and perceived value.


  ### Business Recommendations
  
The findings suggest that the Saas company should focus less on broad acquisition metrics and more on behavioural signals that indicate whether a trial user is likely to become a paying customer:
Prioritise early activation:
The first three days of the trial period are the most important stage of the customer journey. NovaMind should implement guided onboarding, activation checklists and automated product tours that encourage users to experience value immediately after sign-up.

Improve onboarding simplicity:
Medium cognitive load was strongly associated with churn. The onboarding experience should be simplified by reducing unnecessary complexity and guiding users toward a small number of high-impact actions before exposing advanced functionality.

Increase feature adoption:
Users who explored more features converted at much higher rates. NovaMind should use in-app prompts, personalised recommendations and feature discovery campaigns to encourage broader product adoption during the trial period.

Encourage collaborative usage:
Team collaboration was one of the strongest positive conversion signals. The platform should actively encourage users to invite teammates and integrate collaborative workflows, as multi-user adoption increases perceived value and switching costs.

Reduce friction proactively:
Support ticket volume and high complexity scores were associated with lower conversion rates. NovaMind should identify users experiencing repeated friction and provide proactive assistance before trial expiry through live support, onboarding assistance or targeted educational content.

Implement a behavioural resource prioritisation framework:
Customer success and sales resources should be allocated according to behavioural risk segments rather than equally across all trial users.

Risk Segment
Recommended Action
High Risk (0–25)
Minimal human intervention. Use automated nurture emails, self-service onboarding content and low-cost retention efforts, as these users exhibit strong churn signals and low expected return on resource investment.
Medium Risk (26–50)
Provide educational campaigns, onboarding reminders and light-touch engagement. These users show some interest but still experience friction or inconsistent engagement.
Low Risk (51–75)
Prioritise customer success outreach, conversion incentives and product guidance. More than half of users in this segment converted, making them a high-value opportunity group.
Very Likely to Convert (76–100)
Fast-track sales engagement, upselling and expansion opportunities. These users demonstrate strong behavioural commitment and are the highest-probability revenue segment.

This approach allows NovaMind to concentrate human effort on users with the highest expected return while reducing time and cost spent on users who are unlikely to convert.


##  Limitations

The dataset was synthetically generated using Python rather than sourced from a real SaaS company. While behavioural patterns were embedded based on established SaaS benchmarks and psychology research, the findings cannot be directly generalised to real world business contexts without validation against live production data. Also, the analysis was conducted on a fictional single-product SaaS company. Findings may not generalise across different product types, pricing models, or target markets without further validation. 

The synthetic data generation created overly distinct behavioural differences between converted and churned users with insufficient noise and overlap. This manifested in two ways; initial machine learning models achieved perfect AUC scores of 1.0, indicating data leakage rather than genuine predictive power. The project was subsequently reoriented toward a rule-based behavioural scoring approach.  Feature adoption analysis showed 100% conversion rates for users with 7+ features used, because high feature counts were exclusively assigned to converted users during generation. This finding should be interpreted directionally rather than literally. 

The High Cognitive Load tier contained only 39 users (0.78% of the dataset), producing a 0% conversion rate that is statistically unreliable. Conclusions about cognitive load impact are based primarily on Low and Medium tier comparisons which contain adequate sample sizes. 

The conversion risk scoring framework assigns points based on behavioural thresholds derived from this analysis. The weightings; early activation 25 points, onboarding 20 points etc, reflect analytical judgement rather than statistically optimised coefficients. In a production environment, weights would be calibrated using real conversion data and validated through A/B testing. 

Behavioural friction indicators are proxy measures derived from usage data. Direct psychological measurement would require additional user research methods such as surveys, interviews or eye-tracking studies. The frameworks applied;cognitive load theory, present bias, Fogg behaviour model, provide a theoretical lens for interpretation but cannot be formally validated through observational data alone. 


##  Conclusion

This project explored the behavioural patterns that influence a business’ SaaS trial conversion by combining exploratory analysis, behavioural psychology principles, feature engineering, cloud-based data workflows, and rule-based risk scoring. The analysis identified key behavioural signals associated with conversion, including early activation, onboarding completion, feature adoption, engagement depth, collaboration, and recency of activity. By translating these signals into an explainable risk scoring framework, the project provides a practical approach for identifying high-potential users, prioritising customer success efforts, and reducing resource allocation to low-probability prospects. The final solution demonstrates how behavioural data can be transformed into actionable business insights through a modern ELT pipeline using Supabase, Python, and Data studio (looker studio). 
