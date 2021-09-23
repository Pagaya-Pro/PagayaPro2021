from courses.planet_wars.player_bots.ender.EnderBot import EnderBot
from courses.planet_wars.player_bots.fun_with_flags.baseline_bot import NerdBot
from courses.planet_wars.player_bots.rocket_league.baseline_bot import rocket_league_bot
from courses.planet_wars.player_bots.space_pirates.baseline_bot import Firstroundstrategy
from courses.planet_wars.player_bots.under_the_hood.baseline_bot import UnderTheHoodBot
from courses.planet_wars.tournament import Tournament, get_map_by_id
import warnings
import pandas as pd

# Insert Your bot object here, as BotObject(). Don't forget to set BotObject.NAME to your team name
PLAYER_BOTS = [
    Firstroundstrategy(), NerdBot(), Bot1(), EnderBot(), rocket_league_bot(), UnderTheHoodBot(),
]

if __name__ == '__main__':
    # Display options
    warnings.simplefilter(action='ignore')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('expand_frame_repr', False)

    tournament = Tournament(PLAYER_BOTS, ["secret :)"])
    battle_results = tournament.run_tournament()
    player_scores_df = tournament.get_player_scores_data_frame()
    battle_results_df = tournament.get_battle_results_data_frame()
    print(player_scores_df)
    print(battle_results_df)

    player_scores_df.to_parquet("./player_scores_df.parquet")
    battle_results_df.to_parquet("./battle_results_df.parquet")
    player_scores_df.to_csv("./player_scores_df.csv")
    battle_results_df.to_csv("./battle_results_df.csv")
    # TODO commit the saved df so all players can see the battle results