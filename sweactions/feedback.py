from swe_actions import Action, Context

class RequestFeedbackAction(Action):
    def execute(self, context: Context):
        """
        Prompts the user to provide additional information to help complete the task.
        """
        # Prompt the user for feedback
        feedback = input("Please provide more information to complete the task: ")
        # Store the feedback in the context entities
        context.entities['feedback'] = feedback
