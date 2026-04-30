class ActionEngine:
    @staticmethod
    def execute_action(order_data, ai_decision):
        action_taken = "APPROVED"
        
        if "Reject" in ai_decision or "Fraud" in ai_decision:
            action_taken = "ACCOUNT_BLOCKED"
        elif "Flag" in ai_decision or "Review" in ai_decision:
            action_taken = "FUNDS_HELD"
            
        return action_taken