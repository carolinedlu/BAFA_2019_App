import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Findings", page_icon=':mag_right:',layout='wide')
df = pd.read_csv('pages/Data/BAFA_2019_Full.csv',index_col=False)


F_Division = st.container()
F_Team = st.container()

with F_Division:
    st.subheader('Findings by Division')

    # Changing for numeric values in Scores
    for i in range(len(df)):
        if df['Home_Score'].loc[i] == 'A':
            df.at[i,'Home_Score'] = 0
            df.at[i,'Away_Score'] = 50
        elif df['Home_Score'].loc[i] == 'H':
            df.at[i,'Home_Score'] = 50
            df.at[i,'Away_Score'] = 0
    df[['Home_Score']] = df[['Home_Score']].apply(pd.to_numeric)
    df[['Away_Score']] = df[['Away_Score']].apply(pd.to_numeric)
    game_result = []
    total_score = []
    for i in range(len(df)):
        h_s = df['Home_Score'].loc[i]
        a_s = df['Away_Score'].loc[i]
        if h_s > a_s:
            result = 'HW'
        elif h_s < a_s:
            result = 'AW'
        else:
            result = 'T'
        game_result.append(result)
        total_score.append(h_s+a_s)
    df['Game_Result'] = game_result
    df['Total_Score'] = total_score

    options1 = st.selectbox('Choose a Division to check Scores:',tuple(pd.unique(df['Division']).tolist()))


    by_division = df[df.Division == options1]

    by_teams = by_division.groupby('Home_Team').count()
    team_s = [by_teams.iloc[i].name for i in range(len(by_teams))]

    scored_home = []
    scored_away = []
    received_home = []
    received_away = []
    topie = {}
    for team in team_s:
        total_homepoints = sum(by_division['Home_Score'][by_division.Home_Team.str.contains(team)])
        total_awaypoints = sum(by_division['Away_Score'][by_division.Away_Team.str.contains(team)])
        scored_home.append(total_homepoints)
        scored_away.append(total_awaypoints)
        
        total_homereceived = sum(by_division['Away_Score'][by_division.Home_Team.str.contains(team)])
        total_awayreceived = sum(by_division['Home_Score'][by_division.Away_Team.str.contains(team)])
        received_home.append(total_homereceived)
        received_away.append(total_awayreceived)
        topie[team] = [[total_homepoints,total_awaypoints],[total_homereceived,total_awayreceived]]

    colc, cold,  = st.columns([2,2])
    with colc:
        width = 0.5       # the width of the bars: can also be len(x) sequence

        fig, ax = plt.subplots()

        ax.barh(team_s, scored_home,width, label='Score @ Home',color='navy')
        ax.barh(team_s, scored_away, width, left=scored_home,
            label='Score @ Away')

        ax.set_ylabel('Teams')
        ax.set_title('Scores in Favor by Team')
        ax.legend()
        st.pyplot(fig)
    with cold:
        width = 0.5       # the width of the bars: can also be len(x) sequence

        fig1, ax1 = plt.subplots()

        ax1.barh(team_s, received_home,width, label='Received @ Home',color='navy')
        ax1.barh(team_s, received_away, width, left=received_home,
            label='Received @ Away')

        ax1.set_ylabel('Teams')
        ax1.set_title('Scores Against by Team')
        ax1.legend()
        st.pyplot(fig1)

    option_team = st.selectbox('Choose a team for season details',tuple(team_s))

    team_season = df[(df['Away_Team'].str.contains(option_team)) | (df['Home_Team'].str.contains(option_team))]

    game_status = []
    for i in range(len(team_season)):
        if team_season.iloc[i].Home_Team == option_team:
            if team_season.iloc[i].Home_Score > team_season.iloc[i].Away_Score:
                game_status.append('W')
            elif team_season.iloc[i].Home_Score < team_season.iloc[i].Away_Score:
                game_status.append('L')
            else:
                game_status.append('T') 
        elif team_season.iloc[i].Away_Team == option_team:
            if team_season.iloc[i].Home_Score < team_season.iloc[i].Away_Score:
                game_status.append('W')
            elif team_season.iloc[i].Home_Score > team_season.iloc[i].Away_Score:
                game_status.append('L')
            else:
                game_status.append('T') 
    team_season['Game_Status'] = game_status
    team_season['Game_Date'] = pd.to_datetime(team_season['Game_Date'],infer_datetime_format=True)
    team_season = team_season.sort_values(by='Game_Date') # to change the datetype
    ##### new data Frame

    teams_test = df.Home_Team.unique()
    redone = {'Date':[],'Team':[],'Division':[],'Score':[],'Opponent':[],'Travel_km':[],'Game_Status':[],'Team_Status':[],'Score_Type':[],'Temp':[],'Windspeed':[],'Humidity':[],'Visibility':[],
    'Precipitation':[],'Cast':[]}

    for team in teams_test:

        df_test = df[(df.Home_Team == team) | (df.Away_Team == team)]

        for game in range(len(df_test)):
            date = f"{df_test.iloc[game].Game_Date} {df_test.iloc[game].Game_Time}"

            if df_test.iloc[game].Home_Team == team:
                team_score = df_test.iloc[game].Home_Score
                opponent_score = df_test.iloc[game].Away_Score
                score_type = 'Score @ Home'
                travel_distance = 0
                game_status = '@ Home'
                
                if df_test.iloc[game].Home_Score > df_test.iloc[game].Away_Score:
                    team_status = 'W'
                elif df_test.iloc[game].Home_Score < df_test.iloc[game].Away_Score:
                    team_status = 'L'
                else:
                    team_status = 'T'

            
            elif df_test.iloc[game].Away_Team == team:
                team_score = df_test.iloc[game].Away_Score
                opponent_score = df_test.iloc[game].Home_Score
                score_type = "Score @ Away"
                game_status = '@ Away'
                travel_distance = df_test.iloc[game].Travel_km

                if df_test.iloc[game].Home_Score < df_test.iloc[game].Away_Score:
                    team_status = 'W'
                elif df_test.iloc[game].Home_Score > df_test.iloc[game].Away_Score:
                    team_status = 'L'
                else:
                    team_status = 'T'

            redone['Division'].append(df_test.iloc[game].Division)
            redone['Team'].append(team)
            redone['Score'].append(team_score)
            redone['Opponent'].append(opponent_score)
            redone['Score_Type'].append(score_type)
            redone['Travel_km'].append(travel_distance)
            redone['Game_Status'].append(game_status)
            redone['Team_Status'].append(team_status)
            redone['Temp'].append(df_test.iloc[game].Temp)
            redone['Windspeed'].append(df_test.iloc[game].Windspeed)
            redone['Humidity'].append(df_test.iloc[game].Humidity)
            redone['Visibility'].append(df_test.iloc[game].Visibility)
            redone['Cast'].append(df_test.iloc[game].Cast)
            redone['Date'].append(date)
            redone['Precipitation'].append(df_test.iloc[game].Precipitation)

    newmaster = pd.DataFrame.from_dict(redone)
    newmaster['Date'] = pd.to_datetime(newmaster['Date'],infer_datetime_format=True)
    newmaster['Date_day'] = newmaster['Date'].dt.date

    st.table(newmaster[['Date_day','Team','Division','Score','Opponent','Game_Status','Team_Status','Score_Type']][newmaster.Team == option_team].sort_values(by='Date_day'))

    scores_season_team={'Game_Date':[],'Score':[],'Game_Result':[],'Opp_Score':[]}
    for game in range(len(team_season)):
        # print(team_season.iloc[game].Home_Team)
        scores_season_team['Game_Date'].append(team_season.iloc[game].Game_Date)

        if team_season.iloc[game].Home_Team == option_team:
            
            scores_season_team['Score'].append(team_season.iloc[game].Home_Score)
            scores_season_team['Opp_Score'].append(team_season.iloc[game].Away_Score)
            scores_season_team['Game_Result'].append('Score @ Home')
        elif team_season.iloc[game].Away_Team == option_team:
            
            scores_season_team['Score'].append(team_season.iloc[game].Away_Score)
            scores_season_team['Opp_Score'].append(team_season.iloc[game].Home_Score)
            scores_season_team['Game_Result'].append("Score @ Away")

    cole , colf,colg = st.columns([2,2,2])
    with cole:
        st.markdown('######   Scores Ratio')
        vals = np.array(topie[option_team])
        fig2, ax2 = plt.subplots()

        size = .25


        cmap = plt.colormaps["tab20c"]
        outer_colors = cmap(np.arange(2)*4)
        inner_colors = cmap([1,2,5,6])

        ax2.pie(vals.sum(axis=1), radius=.5, colors=outer_colors,labels=['Favor','Against'],autopct='%1.1f%%',
            wedgeprops=dict(width=size, edgecolor='w'))

        ax2.pie(vals.flatten(), radius=1-size, colors=inner_colors,labels=['H','A']*2,
            wedgeprops=dict(width=size, edgecolor='w'))

        ax2.set(aspect='equal', title='Scores ratio by '+option_team)
        st.pyplot(fig2)
    with colf:
        st.markdown('######   Games Played @ Home')
        fig3 = plt.figure()
        sns.countplot(x='Game_Result',data=df[df['Home_Team'].str.contains(option_team)],hue='Cast')
        st.pyplot(fig3)
    with colg:
        st.markdown('######   Games played @ Away')
        fig4 = plt.figure()
        sns.countplot(x='Game_Result',data=df[df['Away_Team'].str.contains(option_team)],hue='Cast')
        st.pyplot(fig4)

    colh, coli = st.columns([1,1])
    with colh:
        st.markdown('######  Overall Game Result in Season')
        fig5 = plt.figure()
        sns.countplot(x='Game_Status',data=team_season,label='Season')
        st.pyplot(fig5)
    with coli:
        st.markdown('######  Game Scores By Team and Opponent By Game')
        fig8, ax8 = plt.subplots(layout='constrained')
        x_games = ['Game %s'%str(i+1) for i in range(len(scores_season_team['Score']))]
        ax8.plot(x_games, scores_season_team['Opp_Score'],'v',markersize=8,color='red',label='Opponent Score')
        ax8.plot(x_games, scores_season_team['Score'],'^',markersize=10,color='blue',label='Team Score')
        ax8.plot(x_games, scores_season_team['Score'],linestyle='dashed',color='blue')
        ax8.plot(x_games, scores_season_team['Opp_Score'],linestyle='dotted',color='red')

        ax8.set_ylabel('Score')
        ax8.set_title('Season by Team')
        ax8.legend()
        plt.xticks(rotation =45)

        st.pyplot(fig8)

   

with F_Team:
    st.subheader('Findings by Team')

