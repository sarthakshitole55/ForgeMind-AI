from app.agents.router import RouterAgent

router = RouterAgent()


def supervisor(state):

    decision = router.route(
        state["question"]
    )

    state["route"] = decision.route

    print("\n========== SUPERVISOR ==========")
    print("Question :", state["question"])
    print("Route    :", decision.route)
    print("Reason   :", decision.reason)
    print("================================\n")

    return state