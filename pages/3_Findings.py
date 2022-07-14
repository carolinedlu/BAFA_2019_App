import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Findings", page_icon=':mag_right:',layout='wide')
df = pd.read_csv('pages/Data/BAFA_2019_complete.csv',index_col=False)

newmaster = pd.read_csv('pages/Data/BAFA_By_team_2019.csv',index_col=False)

F_General = st.container()
F_Division = st.container()
F_Team = st.container()
F_Game = st.container()




with F_General:
    st.subheader('General Findings')
    



with F_Division:
    st.subheader('Findings by Division')

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

    
   

with F_Team:
    st.subheader('Findings by Team')
    option_team = st.selectbox('Choose a team for season details',tuple(team_s))

  
    
    scores_season_team = newmaster[['Date_day','Team','Division','Score','Opponent','Game_Status','Team_Status','Score_Type']][newmaster.Team == option_team].sort_values(by='Date_day')
    st.table(scores_season_team)

    
    cole , colf,colg = st.columns([2,3,3])
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
        sns.countplot(x='Game_Result',data=df[df['Home_Team'].str.contains(option_team)],hue='Cast_home')
        st.pyplot(fig3)
    with colg:
        st.markdown('######   Games played @ Away')
        fig4 = plt.figure()
        sns.countplot(x='Game_Result',data=df[df['Away_Team'].str.contains(option_team)],hue='Cast_home')
        st.pyplot(fig4)

    colh, coli = st.columns([1,1])
    with colh:
        st.markdown('######  Overall Game Result in Season')
        fig5 = plt.figure()
        sns.countplot(x='Team_Status',data=newmaster[(newmaster['Team']==option_team)],hue='Cast')
        st.pyplot(fig5)
    with coli:
        
        st.markdown('######  Game Scores By Team and Opponent By Game')
        fig7, ax7 = plt.subplots(layout='constrained')
        x_games = ['Game %s'%str(i+1) for i in range(len(scores_season_team))]
        ax7.plot(x_games, scores_season_team['Opponent'],'v',markersize=8,color='red',label='Opponent Score')
        ax7.plot(x_games, scores_season_team['Score'],'^',markersize=10,color='blue',label='Team Score')
        ax7.plot(x_games, scores_season_team['Score'],linestyle='dashed',color='blue')
        ax7.plot(x_games, scores_season_team['Opponent'],linestyle='dotted',color='red')

        ax7.set_ylabel('Score')
        ax7.set_title('Season by Team')
        ax7.legend()
        plt.xticks(rotation =45)

        st.pyplot(fig7)



with F_Game:
    st.subheader('Information About Games')

