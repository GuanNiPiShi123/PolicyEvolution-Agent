# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:26:26 2024

@author: yuyajie
"""

import json
import os
import argparse
import random
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
import rlcard
from rlcard import models
from rlcard.utils import set_seed

from setting import Settings, load_model_setting
from agent_model import agi_init, load_llm_from_config
import util

from suspicion_agent import SuspicionAgent 

console = Console()

def check_policy_reserved(game_idx, start_idx, user_index, score, policies):
    if user_index == 1:
        if start_idx == 0:
            if len(policies["first_hand"]["private_policy"])==0 or policies["first_hand"]["private_policy"][-1][1] != game_idx :
                policies["first_hand"]["private_policy"].append(["",game_idx])
            policies["first_hand"]["private_policy"][-1].append(score)
            if policies["first_hand"]["public_policy"]==[] or policies["first_hand"]["public_policy"][-1][1]!=game_idx :
                policies["first_hand"]["public_policy"].append(["",policies["first_hand"]["private_policy"][-1][1], policies["first_hand"]["private_policy"][-1][2]]) 
            else:
                policies["first_hand"]["public_policy"][-1].append(score) 
        else:
            if len(policies["second_hand"]["private_policy"])==0 or policies["second_hand"]["private_policy"][-1][1] != game_idx :
                policies["second_hand"]["private_policy"].append(["",game_idx])
            policies["second_hand"]["private_policy"][-1].append(score)
            if policies["second_hand"]["public_policy"]==[] or policies["second_hand"]["public_policy"][-1][1]!=game_idx:
                policies["second_hand"]["public_policy"].append(["",policies["second_hand"]["private_policy"][-1][1], policies["second_hand"]["private_policy"][-1][2]])
            else:
                policies["second_hand"]["public_policy"][-1].append(score)
    else:
        if start_idx == 1:
            if len(policies["first_hand"]["private_policy"])==0 or policies["first_hand"]["private_policy"][-1][1] != game_idx :
                policies["first_hand"]["private_policy"].append(["",game_idx])
            policies["first_hand"]["private_policy"][-1].append(score)
            if policies["first_hand"]["public_policy"]==[] or policies["first_hand"]["public_policy"][-1][1]!=game_idx :
                policies["first_hand"]["public_policy"].append(["",policies["first_hand"]["private_policy"][-1][1], policies["first_hand"]["private_policy"][-1][2]]) 
            else:
                policies["first_hand"]["public_policy"][-1].append(score) 
        else:
            if len(policies["second_hand"]["private_policy"])==0 or policies["second_hand"]["private_policy"][-1][1] != game_idx :
                policies["second_hand"]["private_policy"].append(["",game_idx])
            policies["second_hand"]["private_policy"][-1].append(score)
            if policies["second_hand"]["public_policy"]==[] or policies["second_hand"]["public_policy"][-1][1]!=game_idx:
                policies["second_hand"]["public_policy"].append(["",policies["second_hand"]["private_policy"][-1][1], policies["second_hand"]["private_policy"][-1][2]])
            else:
                policies["second_hand"]["public_policy"][-1].append(score)
                
    return policies
    

def run(args):
    """
    Run IIG-Policy Evolution
    """
    settings = Settings()
    settings.model = load_model_setting(args.llm)#llm = "openai-gpt-4-0613"

    # Model initialization verification
    res = util.verify_model_initialization(settings)
    if res != "OK":
        console.print(res, style="red")
        return

    # Get inputs from the user
    agent_count = args.agents_num
    agent_configs = []
    if args.load_memory:
        with open(args.memory_record,"r") as m_f:
            m = json.loads(m_f)

    for idx in range(agent_count):
        agent_config = {}
        while True:
            agent_file = args.player1_config if idx == 0 else args.player2_config
            if not os.path.isfile(agent_file):
                console.print(f"Invalid file path: {agent_file}", style="red")
                continue
            try:
                agent_config = util.load_json(Path(agent_file))
                agent_config["path"] = agent_file
                if agent_config == {}:
                    console.print(
                        "Empty configuration, please provide a valid one", style="red"
                    )
                    continue
                if args.load_memory:
                    agent_config["memories"]= m
                break
            except json.JSONDecodeError:
                console.print(
                    "Invalid configuration, please provide a valid one", style="red"
                )
                agent_file = Prompt.ask(
                    "Enter the path to the agent configuration file", default="./agent.json"
                )
                continue
        agent_configs.append(agent_config)
        #agent_config=[{"name": "board_game_expert","age": 27,"personality": "flexible","memories":[],"path":"./person_config/Persuader.json"},
        #{"name": "GoodGuy","age": 27,"personality": "flexible","memories":[],"path":"./person_config/GoodGuy.json"}]

    # Get game rule and observation rule from game config
    while True:
        if not os.path.isfile(args.game_config):
            console.print(f"Invalid file path: {args.game_config}", style="red")
            continue
        try:
            game_config = util.load_json(Path(args.game_config))
            game_config["path"] = args.game_config
            if game_config == {}:
                console.print(
                    "Empty configuration, please provide a valid one", style="red"
                )
                continue
            break
        except json.JSONDecodeError:
            console.print(
                "Invalid configuration, please provide a valid one", style="red"
            )
            game_config = Prompt.ask(
                "Enter the path to the agent configuration file", default="./agent.json"
            )
            continue
    #game_config = {"name": "Leduc Hold'em Poker Limit",
    #"game_rule":" the deck consists only two cards of King, Queen and Jack. two players, only two rounds. only one public hand, two-bet maximum. Raise: first round, put 4 more than opponent, following round, put 8 more than opponent. Call: put same amounts as opponent.
    #1 unit small blind, 2 unit  big blind, and one card, then betting. one public card, then bet again. \n Single Game Win/Draw/Lose Rule: . can also 'fold' \n Whole Game Win/Draw/Lose Rule:. \n Winning Payoff Rule:  . \n Lose Payoff Rule:  . ",
    #"observation_rule": "The observation is dict. observation space: `'raw_legal_actions'. 'hand' .   game_num . all_chips .  rest_chips , opponent_rest_chips . analysis your best combination now and potential combinations in future with probability (number)."
    #"path":"./game_config/leduc_limit.json"}

    #Initialize PEAgent
    ctx = agi_init(agent_configs, game_config, console, settings)
    log_file_name = ctx.robot_agents[(args.user_index+1) % args.agents_num].name+'_vs_'+ctx.robot_agents[(args.user_index ) % args.agents_num].name + '_'+args.rule_model + '_'+args.llm+'_'+args.mode
    #opponent_file_name = "suspicion-agent_log"
    #board_game_expert(self)_vs_GoodGuy(opponent)_cfr_openai-gpt-4-0613_automatic
    #args.user_index=1 means that the second user is opponent,here GoodGuy is opponent which uses cfr model not LLM agent

    #Initialize Environment
    env = rlcard.make('leduc-holdem', config={'seed': args.seed})
    env.reset()
    chips = [100,100]
    #print('./memory_data/'+log_file_name + '_long_memory_summary'+'.json')
    #./memory_data/board_game_expert_vs_GoodGuy_cfr_openai-gpt-4-0613_automatic_long_memory_summary.json

    if args.rule_model == 'cfr':
        rule_model = models.load('leduc-holdem-cfr').agents[0]
    elif args.rule_model == "suspicion-agent":
        rule_model = SuspicionAgent(
                        name=ctx.robot_agents[args.user_index%args.agents_num].name,
                        age=27,
                        rule=game_config["game_rule"],
                        game_name=game_config["name"],
                        observation_rule=game_config["observation_rule"],
                        status="N/A",  
                        llm=load_llm_from_config(ctx.settings.model.llm),                   
                        reflection_threshold=8,
                    )
    else:
        import torch
        rule_model = torch.load(os.path.join('./models', 'leduc_holdem_'+args.rule_model+'_result/model.pth'), map_location='cpu')
        rule_model.set_device('cpu')

    #Game Start...
    #train_threshold = int(args.game_num*0.4)
    #valid_threshold = int(args.game_num*0.5)
    train_threshold = 2
    valid_threshold = 2 + train_threshold
    num_valid = int((valid_threshold-train_threshold)/2)
    valid_batch= 2 
    valid_interval = 1 
    #must satisfy train_threshold/2 /valid_interval == num_valid/2 / valid_batch 
    #evaluation_interval = int(train_threshold/((valid_threshold-train_threshold)/num_valid))

    policies = {"first_hand":{"private_policy":[],"public_policy":[]},"second_hand":{"private_policy":[],"public_policy":[]}}
    best_policy = {"first_hand":{"private_policy":'',"public_policy":''},"second_hand":{"private_policy":'',"public_policy":''}}
    if args.load_policy:
        with open(args.policy_record,"r") as p_f:
            p = json.loads(p_f)
            policies= p["policies"]
            best_policy =p["best_policy"]
    #believes = {ctx.robot_agents[args.user_index].name:[],ctx.robot_agents[(args.user_index+1)%args.agent_num].name:[]}
    #preserve policies and believes of train stage
    if ctx.robot_agents[(args.user_index + 1)%args.agents_num].memory == []:
        ctx.robot_agents[(args.user_index + 1)%args.agents_num].memory = [[],[]]

    valid_id = 0
    
    for game_idx in range(args.game_num):
        
        if game_idx < train_threshold:
            stage = "train"

        elif game_idx < valid_threshold:
            stage = "valid" 
            valid_id+=1
            if game_idx == train_threshold:
                evaluation_flag = 0
                accum_payoff = 0
                train_cores_index = valid_interval-1
                #for name, policy in policies.items():
                best_policy["first_hand"]["private_policy"] = policies["first_hand"]["private_policy"][-1][0]
                best_policy["first_hand"]["public_policy"] = policies["first_hand"]["public_policy"][-1][0]
                best_policy["first_hand"]["score"]=0.0
                best_policy["second_hand"]["private_policy"] = policies["second_hand"]["private_policy"][-1][0]
                best_policy["second_hand"]["public_policy"] = policies["first_hand"]["public_policy"][-1][0]
                best_policy["second_hand"]["score"]=0.0
        else:
            stage = "test"
            
        print("Stage is "+stage+". Game ID is "+str(game_idx+1)+".")

        #reset memory,seed and environment

        #bot_short_memory[0] or bot_short_memory[1] records game_id, self obeservation and action as well as opponents's action at each step, and win message
        #bot_long_memory[0] or bot_long_memory[1] records game_id, self obeservation, policy, belief and action or opponents's observation and actionat each step
        bot_long_memory = []
        bot_short_memory = []
        oppo_bot_short_memory = []
        for i in range(args.agents_num):
            bot_short_memory.append([f'{game_idx+1}th Game Starts.'])
            oppo_bot_short_memory.append([f'{game_idx+1}th Game Starts.'])
            bot_long_memory.append([f'{game_idx+1}th Game Starts.'])
            
        if stage == "test":
            if args.random_seed:
                set_seed(random.randint(0,10000))
            else:
                set_seed(args.seed)
            env.reset()
            
        elif stage == "train":
            if args.random_seed:
                seed = random.randint(0,10000)
                while True:
                    set_seed(seed)
                    env.reset()
                    start_idx = env.get_player_id()
                    if (game_idx%2 == start_idx and args.user_index==1) or ((game_idx+1)%2 == start_idx and args.user_index==0):
                        break
                    else:
                        seed = seed +1                    
            else:
                set_seed(args.seed)
                env.reset()
        else:

            if valid_id<=num_valid:
                if valid_id % valid_batch==1:
                    init_seed = 10                
                while True:
                    set_seed(init_seed)
                    env.reset()
                    start_idx = env.get_player_id()
                    init_seed +=1
                    if start_idx != args.user_index:
                        break
                    
            else:
                if valid_id % valid_batch==1:
                    init_seed = 10 
                while True:
                    set_seed(init_seed)
                    env.reset()
                    start_idx = env.get_player_id()
                    init_seed +=1
                    if start_idx == args.user_index:
                        break
                
        #betting process
        round = 0 # actually it means step, i.e. in Leduc_limit Game, round<4
        policy_index = False #indicates which round policy is best
        public_update = False
        private_update = False
        
        while not env.is_over():
            idx = env.get_player_id()# indicates which palyer should take action
            if round == 0:
                start_idx = idx#indicates which palyer should take action first in the round 0 
                print("start id is :"+str(start_idx))
            if args.user_index == idx and args.user:#args.user_index means opponent index, args.user= true means that opponent uses baseline model such as cfr
                oppo_obs = env.get_state(env.get_player_id())['raw_obs']
                console.print(oppo_obs, style="green")#opponent's observation before take action
                if args.rule_model == "suspicion-agent":                    
                    oppo_obs['game_num'] = game_idx+1
                    oppo_obs['rest_chips'] = chips[1]
                    oppo_obs['opponent_rest_chips'] = chips[0]
                    valid_action_list = env.get_state(env.get_player_id())['raw_legal_actions']
                    #print("valid_action_list is:")
                    #print(valid_action_list)
                    my_agent_name  = ctx.robot_agents[(idx+1)%args.agents_num].name
                    act, oppo_comm, oppo_bot_short_memory, _ = rule_model.make_act(oppo_obs, my_agent_name, env.get_player_id(),valid_action_list, verbose_print= args.verbose_print, game_idx = game_idx,round=round, bot_short_memory=oppo_bot_short_memory, bot_long_memory=bot_long_memory, console=console, log_file_name=None ,mode="first_tom")
                else:
                    act,_ = rule_model.eval_step(env.get_state(env.get_player_id()))
                    act = env._decode_action(act)

                util.get_logging(logger_name = log_file_name + '_opponent_observation',
                    content={str(game_idx + 1) + "_" + str(round+1): {"raw_obs": oppo_obs}})
                    #content={"1_0":{"raw_obs":observation decription after taking ction}}
                util.get_logging(logger_name = log_file_name + '_opponent_action',
                                 content={str(game_idx + 1) + "_" + str(round+1): {
                                     "act": str(act), "talk_sentence": str("")}})
                                 #content={"1_0":{"act": action has been taken, "talk_sentence":""(baseline model is set unalbe to perform ToM talk)}}
                console.print(act, style="green")

                bot_short_memory[(args.user_index) % args.agents_num].append(
                    f"{ctx.robot_agents[args.user_index].name} have the observation: {oppo_obs}, and try to take action: {act}.")
                bot_short_memory[(args.user_index + 1) % args.agents_num].append(
                    f"The valid action list of {ctx.robot_agents[args.user_index].name} is {env.get_state(env.get_player_id())['raw_legal_actions']}, and he tries to take action: {act}.")

                if args.no_hindsight_obs:
                    #do not add the opponent’s observation into the single game history after the end of each game
                    bot_long_memory[(args.user_index) % args.agents_num].append(
                        f"{ctx.robot_agents[args.user_index].name} try to take action: {act}.")
                else:
                    #add the opponent’s observation into the single game history after the end of each game
                    bot_long_memory[(args.user_index) % args.agents_num].append(
                        f"{ctx.robot_agents[args.user_index].name} have the observation: {oppo_obs}, and try to take action: {act}.")
            else:#it is trun of LLM agent to take action.
                if start_idx == args.user_index:
                    ctx.robot_agents[idx].order = "second_hand"
                else:
                    ctx.robot_agents[idx].order = "first_hand"
                amy = ctx.robot_agents[idx]# amy = PEAgent
                amy_index = env.get_player_id()

                amy_obs = env.get_state(env.get_player_id())['raw_obs']
                amy_obs['game_num'] = game_idx+1
                amy_obs['rest_chips'] = chips[0]
                amy_obs['opponent_rest_chips'] = chips[1]

                valid_action_list = env.get_state(env.get_player_id())['raw_legal_actions']
                print("observation is:")
                print(amy_obs)
                #print("valid_action_list is:")
                #print(valid_action_list)
                opponent_name  = ctx.robot_agents[(idx+1)%args.agents_num].name
                #print("opponent_name:"+opponent_name)

                #get old policy
                if stage == "train":
                    if start_idx == args.user_index:
                        if policy_index == True:
                            if len(policies["second_hand"]["public_policy"]) !=0:
                                old_policy = policies["second_hand"]["public_policy"][-1][0]
                            else:
                                old_policy = ""
                        else:
                            if len(policies["second_hand"]["private_policy"]) !=0:
                                old_policy = policies["second_hand"]["private_policy"][-1][0]
                            else:
                                old_policy = ""
                    else:
                        if policy_index == True:
                            if len(policies["first_hand"]["public_policy"]) !=0:
                                old_policy = policies["first_hand"]["public_policy"][-1][0]
                            else:
                                old_policy = ""
                        else:
                            if len(policies["first_hand"]["private_policy"]) !=0:
                                old_policy = policies["first_hand"]["private_policy"][-1][0]
                            else:
                                old_policy = ""
                        
                elif stage == "valid":
                    if valid_id > num_valid:
                        if policy_index == True: 
                            old_policy = policies["second_hand"]["public_policy"][train_cores_index][0]
                        else:
                            old_policy = policies["second_hand"]["private_policy"][train_cores_index][0]
                    else:
                        if policy_index == True: 
                            old_policy = policies["first_hand"]["public_policy"][train_cores_index][0]
                        else:
                            old_policy = policies["first_hand"]["private_policy"][train_cores_index][0]

                else:
                    if start_idx == args.user_index:
                        if policy_index:
                            old_policy = best_policy["second_hand"]["public_policy"]
                        else:
                            old_policy = best_policy["second_hand"]["private_policy"]
                    else:
                        if policy_index:
                            old_policy = best_policy["first_hand"]["public_policy"]
                        else:
                            old_policy = best_policy["first_hand"]["private_policy"]
                #core code: envoke LLM to give the commment, memory and action
                act, comm, bot_short_memory, bot_long_memory, new_policy = amy.make_act(amy_obs, opponent_name, amy_index, valid_action_list, verbose_print= args.verbose_print,
                                                                            game_idx = game_idx, round=round, bot_short_memory=bot_short_memory, bot_long_memory=bot_long_memory, console=console,
                                                                            log_file_name=log_file_name, mode=args.mode, stage = stage, old_policy = old_policy)
                print("action is:")
                print(act)
                oppo_bot_short_memory[(args.user_index) % args.agents_num].append(f"The valid action list of {ctx.robot_agents[(args.user_index+1)%args.agents_num].name} is {valid_action_list}, and he tries to take action: {act}.")
                oppo_bot_short_memory[(args.user_index+1) % args.agents_num].append(
                    f"{ctx.robot_agents[(args.user_index+1)%args.agents_num].name} have the observation: {amy_obs}, and try to take action: {act}.")
                
                #preserve generated policy during train stage
                if stage == "train":                    
                    if start_idx == args.user_index:
                        if policy_index == True and not public_update:
                            policies["second_hand"]["public_policy"].append([new_policy,game_idx])
                            public_update = True
                        elif policy_index == False and not private_update:
                            policies["second_hand"]["private_policy"].append([new_policy,game_idx])
                            private_update = True
                    else:
                        if policy_index == True and not public_update:
                            policies["first_hand"]["public_policy"].append([new_policy,game_idx])
                            public_update = True
                        elif policy_index == False and not private_update:
                            policies["first_hand"]["private_policy"].append([new_policy,game_idx])
                            private_update = True
                else:
                    pass
                
                if amy_obs["public_card"]!=None:
                    policy_index = True

            env.step(act, raw_action=True)
            round += 1
            
        private_update=False
        public_update = False
        
        #update chips after each game
        pay_offs = env.get_payoffs()
        
        if stage == "train":
            if (args.user_index+1)%args.agents_num == 0:
                score = pay_offs[0]
            else:
                score = pay_offs[1]
            policies = check_policy_reserved(game_idx, start_idx, args.user_index, score, policies)
        
        
        if stage == "test":
            for idx in range(len(pay_offs)):
                if args.user_index == 0:
                    chips[(idx+1)%args.agents_num] += pay_offs[idx]
                else:
                    chips[idx] += pay_offs[idx]
        if stage == "valid":
            evaluation_flag += 1#the accumlated game number
            if (args.user_index+1)%args.agents_num == 0:
                accum_payoff += pay_offs[0]
            else:
                accum_payoff += pay_offs[1]

            if evaluation_flag == valid_batch:#judge whether to update best policy
                if valid_id <= num_valid:
                    if accum_payoff/valid_batch >  best_policy["first_hand"]["score"]:    
                        best_policy["first_hand"]["score"]=accum_payoff#judge if to update best policy
                        best_policy["first_hand"]["private_policy"] = policies["first_hand"]["private_policy"][train_cores_index][0]
                        best_policy["first_hand"]["public_policy"] = policies["first_hand"]["public_policy"][train_cores_index][0]
                else:
                    if accum_payoff/valid_batch >  best_policy["second_hand"]["score"]:
                        best_policy["second_hand"]["score"]=accum_payoff
                        best_policy["second_hand"]["private_policy"] = policies["second_hand"]["private_policy"][train_cores_index][0]
                        best_policy["second_hand"]["public_policy"] = policies["second_hand"]["public_policy"][train_cores_index][0]
                print("train_cores_index is")
                print(train_cores_index)
                print("score is")
                print(accum_payoff/valid_batch)
                train_cores_index = +valid_interval
                if valid_id == num_valid:
                    train_cores_index = valid_interval-1
                evaluation_flag = 0
                accum_payoff = 0

        print("pay_offs:")
        print(pay_offs)
        if pay_offs[0] > 0:
            win_message = f'{ctx.robot_agents[0].name} win {pay_offs[0]} chips, {ctx.robot_agents[1].name} lose {pay_offs[0]} chips.'
        else:
            win_message = f'{ctx.robot_agents[1].name} win {pay_offs[1]} chips, {ctx.robot_agents[0].name} lose {pay_offs[1]} chips.'
        #print(win_message)
        
        #bot_short_memory[0].append(win_message)
        #bot_short_memory[1].append(win_message)
        #bot_long_memory[0].append(win_message)
        #bot_long_memory[1].append(win_message)

        #the first element is history list of position_0 user, the second element is history list of position_1 user
        #the short history list example of opponent: ['1th Game Start', 'GoodGuy have the observation:xxx, and try to take action:xxx',
        #board_game_expert try to take action: xxx and say xxx to GoodGuy, ...(echange the turn)]
        #the short history list example of PEAgent: ['1th Game Start', 'The valid action list of GoodGuy is:raw_legal_actions,, and he tries to take action: {act}.',
        #board_game_expert have the observation xxx, try to take action:xxx and say xxx to GoodGuy, ...(echange the turn)]
        #the long history list example of opponent: ['1th Game Start', 'GoodGuy have the observation':xxx(determined by args.no_hindsight_obs), and try to take action:xxx', "GoodGuy ...", win_message]
        #the long history list example of PEAgent: ['1th Game Start', 'board_game_expert have the observation xxx, try to take action: xxx and say xxx to GoodGuy', "board_game_expert ...",win_message]

        long_memory = '\n'.join(
            [x + '\n' + y for x, y in zip(bot_long_memory[start_idx][1:], bot_long_memory[(start_idx+1)%args.agents_num][1:])])

        #At the end of the game, to get summariztion and reflextion about the game and at the meantime to preserve policies and best policy
        if stage == "train":
            long_memory = f'Here is one of game histories: Game Starts. ' + long_memory + win_message

            #rewrite long history and get reflextion of self and opponent.
            memory_summarization= ctx.robot_agents[(args.user_index + 1) % args.agents_num].get_summarization(ctx.robot_agents[(args.user_index+1) % args.agents_num].name,
                long_memory, ctx.robot_agents[args.user_index].name, no_highsight_obs = args.no_hindsight_obs)
            #In the first round of first game, name holds card1 does action .... continue ..."

            ctx.robot_agents[(args.user_index + 1) % args.agents_num].add_long_memory("One of Games Started! " + memory_summarization, start_idx == args.user_index)

            
            util.get_logging(logger_name= log_file_name + '_long_memory',
                            content={str(game_idx + 1): {"long_memory": long_memory}})
            util.get_logging(logger_name= log_file_name + '_long_memory_summary',
                        content={str(game_idx + 1): {"long_memory_summary": memory_summarization}})

        elif stage == "test":
            long_memory = f'{game_idx+1-valid_threshold}th Game Start! \n' + long_memory + win_message
            memory_summarization= ctx.robot_agents[(args.user_index + 1) % args.agents_num].get_summarization(ctx.robot_agents[(args.user_index+1) % args.agents_num].name,
                long_memory, ctx.robot_agents[(args.user_index) % args.agents_num].name, no_highsight_obs = args.no_hindsight_obs)

            ctx.robot_agents[(args.user_index + 1)%args.agents_num].add_long_memory('One of Games Started! ' + memory_summarization, start_idx == args.user_index)

            util.get_logging(logger_name= log_file_name + '_long_memory',
                                content={str(game_idx + 1): {"long_memory": long_memory}})
            util.get_logging(logger_name= log_file_name + '_long_memory_summary',
                                content={str(game_idx + 1): {"long_memory_summary": memory_summarization}})

        else:
            pass
        
    f1 = open(args.policy_record,"w")
    f1.write(json.dumps({"policies":policies,"best_policy":best_policy}, ensure_ascii =False))
    f1.close()

    f2 = open(args.memory_record,"w")
    f2.write(json.dumps(ctx.robot_agents[(args.user_index+1) % args.agents_num].memory, ensure_ascii =False))
    f2.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Policy Evolution Agent',
        description='Playing Imperfect Information Games with LLM Based on Policy Evolution',
        epilog='Text at the bottom of help')

    parser.add_argument("--player1_config", default="./person_config/Persuader.json", help="experiments name")
    parser.add_argument("--player2_config", default="./person_config/GoodGuy.json", help="experiments name")
    parser.add_argument("--game_config", default="./game_config/leduc_limit.json",
                        help="./game_config/leduc_limit.json, ./game_config/limit_holdem.json, ./game_config/coup.json")

    parser.add_argument("--seed", type=int, default=1, help="random_seed")

    parser.add_argument("--llm", default="openai-gpt-4-0125", help="environment flag, openai-gpt-4-0125 or openai-gpt-3.5-turbo")
    parser.add_argument("--rule_model", default="cfr", help="rule model: cfr or nfsp or dqn or dmc or suspicion-agent")
    parser.add_argument("--mode", default="first_tom", help="inference mode: normal or first_tom or second_tom or automatic")
    #parser.add_argument("--stage", default="train", help="train or valid or test stage")

    #user stands for opponent
    parser.add_argument("--agents_num", type=int, default=2)
    parser.add_argument("--user", default=True, help="one of the agents is baseline mode, e.g. cfr, nfsp")
    parser.add_argument("--verbose_print", action="store_true",help="""The retriever to fetch related memories.""")
    parser.add_argument("--user_index", type=int, default=1, help="user position: 0 or 1")
    parser.add_argument("--game_num", type=int, default=14)#train:40 valid:10 test:50 每8条一训练，2条进行验证
    parser.add_argument("--random_seed", default=True)
    parser.add_argument("--no_hindsight_obs", default=False, help = "indicates whether to add the opponent’s observation into the single game history after the end of each game")

    parser.add_argument("--load_policy", default = False)
    parser.add_argument("--load_memory", default = False)
    parser.add_argument("--policy_record", default="./leduc_limit_cfr_policy.json", help="experiments name")
    parser.add_argument("--memory_record", default="./leduc_limit_cfr_memory.json", help="experiments name")

    args = parser.parse_args()
    run(args)