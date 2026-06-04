import heapq
import math
import numpy as np
import pandas as pd


# --------------------------------------------------
# Input values from the report
# --------------------------------------------------

SERVICE_MEAN = 299.6        # seconds, approximately 5 minutes
PATIENCE_TIME = 60          # customers abandon after 60 seconds
BREAK_TIME = 50            # seconds, short pause/wrap-up after each call
DAYS_PER_MONTH = 22         # approximate number of weekdays in a month

# Number of replications for the simulation experiment
REPLICATIONS = 300

# Base seed used to make results reproducible
BASE_SEED = 12345

AGENT_SCENARIOS = [4, 5, 6, 7, 8]

MONTH_FACTORS = {
    "October": 1.158,
    "November": 1.29,
    "December": 1.45
}

# Average calls per hour from Figure 4.2 in the report
ARRIVAL_RATES_PER_HOUR = {
    8: 20.8,
    9: 19.5,
    10: 19.5,
    11: 19.6,
    12: 20.1,
    13: 20.1,
    14: 20.2,
    15: 20.3,
    16: 20.4,
    17: 17.9
}


# --------------------------------------------------
# Generate all call arrival events for one day
# --------------------------------------------------

def create_arrival_events(month_factor):
    event_list = []
    event_id = 0
    call_id = 0

    first_hour = min(ARRIVAL_RATES_PER_HOUR.keys())

    for hour, hourly_rate in ARRIVAL_RATES_PER_HOUR.items():

        # Adjust the hourly arrival rate by month factor only
        adjusted_rate = hourly_rate * month_factor

        # Number of calls in the hour follows a Poisson distribution
        number_of_calls = np.random.poisson(adjusted_rate)

        for i in range(number_of_calls):

            # Place each call randomly within the hour
            arrival_time = (hour - first_hour) * 3600 + np.random.uniform(0, 3600)

            # Add the arrival event to the event list
            heapq.heappush(event_list, (arrival_time, event_id, "arrival", call_id))

            event_id += 1
            call_id += 1

    return event_list, event_id


# --------------------------------------------------
# Simulate one workday
# --------------------------------------------------

def simulate_one_day(number_of_agents, month_factor):

    # Event list contains all future events
    event_list, event_id = create_arrival_events(month_factor)

    # State variables
    clock = 0
    busy_agents = 0
    queue = []

    # Information about each call
    calls = {}

    # Output counters
    total_calls = 0
    answered_calls = 0
    abandoned_calls = 0
    answered_within_60_sec = 0

    waiting_times = []
    total_busy_time = 0

    # --------------------------------------------------
    # Start service for a call
    # --------------------------------------------------

    def start_service(call_id, current_time):
        nonlocal busy_agents
        nonlocal answered_calls
        nonlocal answered_within_60_sec
        nonlocal total_busy_time
        nonlocal event_id

        call = calls[call_id]

        # If the call already abandoned, it cannot start service
        if call["abandoned"]:
            return

        call["answered"] = True
        call["service_start"] = current_time

        waiting_time = current_time - call["arrival_time"]
        waiting_times.append(waiting_time)

        answered_calls += 1

        if waiting_time <= 60:
            answered_within_60_sec += 1

        # B = B + 1
        busy_agents += 1

        # Generate service time
        service_time = np.random.exponential(SERVICE_MEAN)

        # Add 30 seconds pause/wrap-up after the call
        total_occupied_time = service_time + BREAK_TIME

        # The agent is first available again after service + break time
        service_end_time = current_time + total_occupied_time

        # Busy time includes both call handling and the short pause
        total_busy_time += total_occupied_time

        heapq.heappush(event_list, (service_end_time, event_id, "service_end", call_id))
        event_id += 1

    # --------------------------------------------------
    # Main discrete-event simulation loop
    # --------------------------------------------------

    while len(event_list) > 0:

        # Move the simulation clock to the next event
        clock, _, event_type, call_id = heapq.heappop(event_list)

        # -----------------------------
        # Event: Call Arrival
        # -----------------------------

        if event_type == "arrival":
            total_calls += 1

            calls[call_id] = {
                "arrival_time": clock,
                "answered": False,
                "abandoned": False,
                "service_start": None
            }

            # If B < C, the call starts service immediately
            if busy_agents < number_of_agents:
                start_service(call_id, clock)

            # If B = C, the call enters the queue
            else:
                queue.append(call_id)

                # Schedule abandonment after patience time
                abandon_time = clock + PATIENCE_TIME
                heapq.heappush(event_list, (abandon_time, event_id, "abandon", call_id))
                event_id += 1

        # -----------------------------
        # Event: Abandon
        # -----------------------------

        elif event_type == "abandon":
            call = calls[call_id]

            # The customer only abandons if service has not started
            if not call["answered"] and not call["abandoned"]:
                call["abandoned"] = True
                abandoned_calls += 1

                # Q = Q - 1
                if call_id in queue:
                    queue.remove(call_id)

        # -----------------------------
        # Event: End Service + Break
        # -----------------------------

        elif event_type == "service_end":

            # B = B - 1
            # The agent becomes available again after both service and break time
            busy_agents -= 1

            # If Q > 0, the first customer in the queue starts service
            if len(queue) > 0:
                next_call_id = queue.pop(0)
                start_service(next_call_id, clock)

    # --------------------------------------------------
    # Calculate performance measures
    # --------------------------------------------------

    if total_calls > 0:
        service_level = answered_within_60_sec / total_calls
        abandonment_rate = abandoned_calls / total_calls
    else:
        service_level = 0
        abandonment_rate = 0

    if len(waiting_times) > 0:
        average_waiting_time = np.mean(waiting_times)
    else:
        average_waiting_time = 0

    opening_time = len(ARRIVAL_RATES_PER_HOUR) * 3600
    agent_utilization = total_busy_time / (number_of_agents * opening_time)

    return {
        "total_calls": total_calls,
        "answered_calls": answered_calls,
        "abandoned_calls": abandoned_calls,
        "service_level": service_level,
        "abandonment_rate": abandonment_rate,
        "average_waiting_time_sec": average_waiting_time,
        "agent_utilization": agent_utilization
    }


# --------------------------------------------------
# Run one replication for one scenario
# --------------------------------------------------

def run_one_replication(month_factor, number_of_agents):
    daily_results = []

    for day in range(DAYS_PER_MONTH):
        one_day_result = simulate_one_day(number_of_agents, month_factor)
        daily_results.append(one_day_result)

    df = pd.DataFrame(daily_results)

    return {
        "service_level": df["service_level"].mean(),
        "abandonment_rate": df["abandonment_rate"].mean(),
        "average_waiting_time_sec": df["average_waiting_time_sec"].mean(),
        "agent_utilization": df["agent_utilization"].mean(),
        "average_calls_per_day": df["total_calls"].mean()
    }


# --------------------------------------------------
# Run the full experiment with 30 replications
# --------------------------------------------------

def run_experiment():
    results = []

    for month_index, (month, month_factor) in enumerate(MONTH_FACTORS.items()):

        for agents in AGENT_SCENARIOS:

            replication_results = []

            for replication in range(REPLICATIONS):

                # New seed for each replication
                seed = BASE_SEED + month_index * 10000 + agents * 1000 + replication
                np.random.seed(seed)

                result = run_one_replication(month_factor, agents)
                result["replication"] = replication + 1
                replication_results.append(result)

            df = pd.DataFrame(replication_results)

            results.append({
                "month": month,
                "agents": agents,
                "service_level": df["service_level"].mean(),
                "abandonment_rate": df["abandonment_rate"].mean(),
                "average_waiting_time_sec": df["average_waiting_time_sec"].mean(),
                "agent_utilization": df["agent_utilization"].mean(),
                "average_calls_per_day": df["average_calls_per_day"].mean()
            })

    return pd.DataFrame(results)


# --------------------------------------------------
# Replication analysis for one selected scenario
# --------------------------------------------------

def replication_analysis(month="December", agents=4, pilot_replications=30):
    month_factor = MONTH_FACTORS[month]
    month_index = list(MONTH_FACTORS.keys()).index(month)

    pilot_results = []

    for replication in range(pilot_replications):

        # New seed for each pilot replication
        seed = BASE_SEED + 50000 + month_index * 10000 + agents * 1000 + replication
        np.random.seed(seed)

        result = run_one_replication(month_factor, agents)
        result["replication"] = replication + 1
        pilot_results.append(result)

    df = pd.DataFrame(pilot_results)

    mean_service_level = df["service_level"].mean()
    std_service_level = df["service_level"].std(ddof=1)

    # t-value for 95% CI with 30 replications is approximately 2.045
    t_value = 2.045

    observed_half_width = t_value * std_service_level / math.sqrt(pilot_replications)
    goal_half_width = 0.05 * mean_service_level

    required_replications = math.ceil((t_value * std_service_level / goal_half_width) ** 2)

    table = pd.DataFrame([{
        "KPI": "Service level",
        "Mean": mean_service_level,
        "Std Dev": std_service_level,
        "Observed half-width": observed_half_width,
        "Half-width %": observed_half_width / mean_service_level * 100,
        "Required replications": required_replications
    }])

    return table


# --------------------------------------------------
# Run model
# --------------------------------------------------

# Make sure Python prints all columns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

results = run_experiment()

# Convert to percentages for readability
results["service_level_%"] = results["service_level"] * 100
results["abandonment_rate_%"] = results["abandonment_rate"] * 100
results["agent_utilization_%"] = results["agent_utilization"] * 100

print("\nSimulation results based on 30 replications:")
print(results[[
    "month",
    "agents",
    "service_level_%",
    "abandonment_rate_%",
    "average_waiting_time_sec",
    "agent_utilization_%",
    "average_calls_per_day"
]])

print("\nMinimum number of agents needed to reach 90% service level:")

for month in MONTH_FACTORS.keys():

    month_results = results[results["month"] == month]
    accepted = month_results[month_results["service_level"] >= 0.90]

    if len(accepted) > 0:
        best = accepted.sort_values("agents").iloc[0]
        print(month, ":", int(best["agents"]), "agents")
    else:
        print(month, ": more than 8 agents may be needed")


print("\nReplication analysis based on pilot run:")
rep_table = replication_analysis(month="December", agents=4, pilot_replications=30)

# Convert the table to easier percentages
rep_table["Mean %"] = rep_table["Mean"] * 100
rep_table["Std Dev %"] = rep_table["Std Dev"] * 100
rep_table["Observed half-width % points"] = rep_table["Observed half-width"] * 100

print(rep_table[[
    "KPI",
    "Mean %",
    "Std Dev %",
    "Observed half-width % points",
    "Half-width %",
    "Required replications"
]])