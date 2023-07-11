# Context22
Presentation of the different parts of the OpenSource project

The project Context22 is made of 2 main parts :
- Assistance to Guardium management : Guardium Add-on
- Database Activity Monitoring (DAM) Security Analytics


## Guardium Add-on
The Add-on to Guardium is a set of features to provide centralized and efficient visibility on the Performance Guardium logs:
- Enrichment and Visualization of the Guardium Report "Buffer Usage monitor" in repository [CT22_EK_Buff](https://github.com/Fredo68usa/CT22_EK_Buff). This provides meaningful display of performance information for the appliances
- RestAPI : The Guardium RestAPI is convenient but requires to develop the client side. This is the client-side, written in Python and allowing you to execute automatically <b>most of the CLI and grdapi commands</b> as well as generating automatically <b>reports</b>. In repository [Context_22_GuardiumRestAPI](https://github.com/Fredo68usa/Context_22_GuardiumRestAPI)
- We posted also some educational presentations on Guardium in the repository : [CT22_Guardium](https://github.com/Fredo68usa/CT22_Guardium) 


## DAM Security Analytics
This track is the heart of the project. The Guardium Add-on is to free companies from the tidious and time consuming work of managing the appliances to free resources for processing collected data for security. This track is made of 3 parts :

### 1 / Enrichment of the DAM Data
This function is crucial to allow for Exploratory Data Analysis. Without it, it is not possible to analyze and understand the Database Traffic. Enrichment adds metadata like server, client and user information as well as labelling of SQLs by types. This piece is done on the Full SQL Entity of Guardium. Repository : [CT22_EK_FullSQL](https://github.com/Fredo68usa/CT22_EK_FullSQL)

### 2 / Predictions of Extractions
The main concern in Database security is to prevent execessive extraction of sensitive data. EDA allows for selecting the extractions SQLs) to be watched and the overall volumes. From those Time series, we generate predictions that can be checked in RT at the time of exrtraction. This is a major function for detecting data leaks as all leaks start in the databases. Currently the prediction is generated using Holt-Winters. Will be posted soon in repository : [CT22_EK_HWES3MUL](https://github.com/Fredo68usa/CT22_EK_HWES3MUL)

### 3 / SQL injection detection
SQL injections are usually detected at the Web Server level. However, since SQL injections are an alteration of an existing SQL, not a brand new one, we look for similarity in SQLs using Fuzzy Logic for such detection. In repository : [CT22_EKP_Fuzzy](https://github.com/Fredo68usa/CT22_EKP_Fuzzy)

## SonarG
We posted a document to set-up the ldaps interface at : [CT22_SonarG](https://github.com/Fredo68usa/CT22_SonarG)

# Contacts
Frederic Petit - email : frederic.guy.petit@gmail.com or fred@context22.com

