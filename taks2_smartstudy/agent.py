from google.adk.agents import (
    Agent,
    SequentialAgent,
    ParallelAgent,
    LoopAgent,
)
from google.adk.tools.tool_context import ToolContext

def exit_loop(tool_context: ToolContext) -> dict:
    """Stop the review loop when the study plan is satisfactory."""
    tool_context.actions.escalate = True
    return {}

def save_review_feedback(
    feedback: str,
    tool_context: ToolContext,
) -> dict:
    """Save review feedback for the next study-plan revision."""
    tool_context.state["review_feedback"] = feedback
    return {"status": "Review feedback saved."}

coordinator = Agent(
    name="coordinator",
    model="gemini-3.8-flash",
    description="Understands the student's study-planning request.",
    instruction="""
    You are the Coordinator for a SmartStudy planning system.

    Read the user's request and identify:
    - subject
    - number of days available
    - topics to study
    - current knowledge level
    - available study time per day

    Extract only information supported by the user's request.
    If information is missing, mark it as unspecified rather
    than inventing an answer.

    Return the extracted information in a clear, structured format.
    """,
    output_key="study_requirements",
)



planning_agent = Agent(
    name="planning_agent",
    model="gemini-3.5-flash",
    description="Creates the overall strategy for a personalized study plan.",
    instruction="""
    You are the Planning Agent.

    Read the study requirements extracted by the Coordinator.

    Study requirements:
    {study_requirements}

    Develop an overall strategy for the study plan using the
    subject, available days, topics, knowledge level, and daily
    study time.

    Do not invent missing details. Clearly identify any
    assumptions or information that remains unspecified.
    """,
    output_key="planning_strategy",
)


topic_agent = Agent(
    name="topic_agent",
    model="gemini-3.5-flash",
    description="Organizes study topics into a logical learning order.",
    instruction="""
    You are the Topic Agent for a SmartStudy planner.

    Use the study requirements from the Coordinator:
    {study_requirements}

    Organize the requested topics in a logical learning order.
    Identify prerequisites where relevant.
    Adapt the ordering to the student's knowledge level.

    Do not invent topics that the user did not request.
    If the topics are unspecified, clearly say so and suggest
    that the user provide a topic list.
    """,
    output_key="topic_analysis",
)


practice_agent = Agent(
    name="practice_agent",
    model="gemini-3.5-flash",
    description="Creates practice questions and exercises for the study plan.",
    instruction="""
    You are the Practice Agent for a SmartStudy planner.

    Use the student's study requirements:
    {study_requirements}

   
    Create suitable practice activities for the requested topics
    and the student's current knowledge level.

    Include a mix of:
    - Conceptual questions
    - Practical exercises
    - Revision or self-assessment questions

    Adjust difficulty to the student's level.
    Do not claim that questions are from an official exam.
    If topics are unspecified, explain that practice activities
    depend on receiving a topic list.

    Return the practice plan in a clear, organized format.
    """,
    output_key="practice_plan",
)


schedule_agent = Agent(
    name="schedule_agent",
    model="gemini-3.5-flash",
    description="Creates a realistic day-by-day study schedule.",
    instruction="""
    You are the Schedule Agent for a SmartStudy planner.

    Use the student's study requirements:
    {study_requirements}

    Use the overall planning strategy if available:
    {planning_strategy}

    Create a day-by-day study schedule that respects:
    - The number of days available
    - The student's daily study time
    - The requested topics
    - The student's current knowledge level

    Allocate realistic time for learning, practice, and revision.
    Do not schedule more study time than the student has available.
    Do not invent missing requirements; identify what is unspecified.

    Return the schedule in a clear, organized format.
    """,
    output_key="schedule",
)


parallel_analysis = ParallelAgent(
    name="parallel_analysis",
    sub_agents=[
        topic_agent,
        practice_agent,
        schedule_agent,
    ],
)


study_plan_writer = Agent(
    name="study_plan_writer",
    model="gemini-3.5-flash",
    description="Combines the planning, topic, practice, and schedule outputs into one study plan.",
    instruction="""
    You are the Study Plan Writer.

    Create a personalized study plan using the following
    information from session state:

    Student requirements:
    {study_requirements}

    Overall planning strategy:
    {planning_strategy}

    Topic analysis:
    {topic_analysis}

    Practice activities:
    {practice_plan}

    Schedule:
    {schedule}
    

    Combine these results into one clear, consistent study plan.

    Include:
    1. The student's requirements
    2. Topics in a logical learning order
    3. A day-by-day schedule
    4. Practice activities
    5. Revision and self-assessment

    Respect the student's available days and daily study time.
    Do not invent missing requirements.
    Identify any missing information or inconsistencies.
    Address every issue marked NEEDS_REVISION.
    Do not change parts of the plan that already satisfy the requirements.

    If previous review feedback is available:
    - If it says NEEDS_REVISION, revise the plan to address the feedback.
    - If it says SATISFACTORY, preserve the approved plan.
    On the first iteration, create the plan normally if no review feedback is available.
    Return the complete study plan in a readable format.
    """,
    output_key="draft_plan",
)


review_agent = Agent(
    name="review_agent",
    model="gemini-3.5-flash",
    description="Reviews the draft study plan for completeness and consistency.",
    tools=[save_review_feedback, exit_loop],
    instruction="""
    You are the Review Agent.

    Review the draft study plan:
    {draft_plan}

    Check whether the plan:
    1. Matches the student's stated requirements.
    2. Covers the requested topics.
    3. Fits within the available number of days.
    4. Respects the daily study-time limit.
    5. Includes practice and revision activities.

    Identify specific problems and explain how to fix them.

    If the plan satisfies all five checks:
    1. Clearly state SATISFACTORY.
    2. Call the exit_loop tool to stop the review loop.

    Otherwise, clearly state:
    NEEDS_REVISION
    
    If the plan needs revision:
    1. Clearly state NEEDS_REVISION.
    2. Explain the specific problems and fixes.
    3. Call save_review_feedback with your feedback.

    If the plan is satisfactory:
    1. Clearly state SATISFACTORY.
    2. Call save_review_feedback with the approval feedback.
    3. Call exit_loop to stop the review loop.""",
    output_key="review_result",
)


review_loop = LoopAgent(
    name="review_loop",
    sub_agents=[
        study_plan_writer,
        review_agent,
    ],
    max_iterations=2,
)


root_agent = SequentialAgent(
    name="smartstudy_workflow",
    sub_agents=[
        coordinator,
        planning_agent,
        parallel_analysis,
        review_loop,
    ],
)
