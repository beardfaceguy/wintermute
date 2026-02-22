# Source: https://temporal.io/blog/building-an-agentic-system-thats-actually-production-ready

- Platform Overview How Temporal Works Temporal Cloud Security

- Overview

- How Temporal Works

- Temporal Cloud

- Security

- Platform

- Docs

- Pricing

- Use Cases Customer Stories AI Financial Services Platform Engineering

- Customer Stories

- AI

- Financial Services

- Platform Engineering

- Use Cases

- Resources Resource Library Learn Temporal Community Code Exchange Blog For Startups Partners Change Log

- Resource Library

- Learn Temporal

- Community

- Code Exchange

- Blog

- For Startups

- Partners

- Change Log

- Resources

- Replay 2026

- Blog

- Temporal Voices

# Building an agentic system that’s actually production-ready

You’re in the industry and your hands are on the keyboard. I know you know that every company is clamoring to ship an AI assistant and that many fall flat on their faces before they even reach the finish line.

Right now, I’ve noticed there’s a massive gap between the exciting potential of agentic AI (conversation, proactive, and helpful) and the reality of building something durable enough to survive production. Everyone’s flaunting their GPT-powered demos, but few have something that still works even two days later.

If you’ve already read this post on why agentic systems are just distributed systems in disguise, this is the follow up. Not the why , but the how .

That’s what inspired me to write this article. Not to further hype the trend, but to strip things down to the foundation (definitions, architecture, and flows) to make sure you’re building on solid ground.

If you’re an engineer, architect, or technical lead thinking about how to scale agent workflows, I’m talking to you!

Since agentic AI is a hot topic now, and there are a lot of terms, ideas, and technologies being discussed, along with the questions:

- Why is this space so interesting? What’s so exciting (and fun!) about it?

- What do all of these terms mean? Is there a consistent definition?

- What does an Agentic System actually do?

- How does Temporal (the technology) help in this space?

- How is Temporal (the company) helping out companies implement these kinds of systems?

This blog aims to define some terms, share example architecture and code samples, and offer some helpful opinions on where this is all headed.

If you’re especially curious about how we as a company are helping, check out these awesome stories and posts:

- Gorgias Uses AI Agents to Improve Customer Service

- Temporal Use Case Round-Up: Generative AI

- How Dust Is Building the Future of Work With Agentic AI and Why it Runs on Temporal

- Building, launching, and scaling ChatGPT Images

## Agentic systems: Potential #

Agentic systems have the potential to redefine application architecture because of their amazing capabilities. Agentic Systems can give you an army of qualified assistants that can analyze, execute, and recommend at a massive scale. They can replace a complex UI presenting hundreds of data points with a simple conversation, highlighting points of interest and massively reducing cognitive load. They can make recommendations and take action in context , dramatically reducing the burden of creation. They can validate and make judgements, reducing risk and building confidence.

In terms of the future of agentic AI, I (and my colleagues here at Temporal across Engineering, Solutions Architecture and GTM, Product, and Developer Relations) expect these systems may replace entire applications, in the same way the smart phone replaced CDs players, paper calendars, desk phones, personal computers, VCRs, and calling people on the phone to get food delivered . I expect agentic systems will also enable entirely new kinds of applications, just like other technology revolutions did.

If these systems can deliver on their promises, being a human connected to the internet is about to get a lot easier.

## Agentic systems: Definitions #

There are lots of terms floating around in AI. Here’s how I define the foundational elements of agentic AI Systems.

- AI : In agentic systems, a Large Language Model (LLM) that simulates intelligence. It can recommend how to pursue goals, make decisions, and recommend or decide on actions to take.

- Goal : Something the AI and user both want to finish. Often executed by one or more tools with the steps determined by the AI.

- Tool : A capability of an agentic system. It performs Actions to be taken to accomplish goals with context to guide the AI about how and when to use the tool. Often implemented as a simple function making an API call. Sometimes tools have multiple steps or their own intelligence (see Agents as Tools, below). Can interact with public and private data sources or APIs. Can be built for a specific agentic system. Can be MCP tools . MCP tools provide their own context that the AI can use to evaluate their usage for the user’s goals.

- It performs Actions to be taken to accomplish goals with context to guide the AI about how and when to use the tool.

- Often implemented as a simple function making an API call.

- Sometimes tools have multiple steps or their own intelligence (see Agents as Tools, below).

- Can interact with public and private data sources or APIs.

- Can be built for a specific agentic system.

- Can be MCP tools . MCP tools provide their own context that the AI can use to evaluate their usage for the user’s goals.

- MCP tools provide their own context that the AI can use to evaluate their usage for the user’s goals.

- User : A person who wants something done.

- Agent : Someone or something that acts on behalf and for the benefit of someone or something else.

- Agentic : An attribute of a system, in which an AI acts on behalf of a user, completing tasks, making decisions, and recommendations for next steps.

- Agentic system : An application that interacts with a user and behaves in an agentic way, using AI and tools to accomplish goals.
    Here is a simple model putting all of these concepts together:

### Advanced agentic systems #

The following concepts are an expansion of those above. We are seeing systems being built that have more complex agentic capabilities:

- Multi-Agent : An attribute of a system that uses multiple agents. Agents could have different capabilities and roles. Agents could be selected via agent routing or tasks could be delegated to agents, for example by implementing an agent as a tool.

- Agents as tools : in a multi-agent system, Agents can be implemented as tools — a capability of the system. Agents can be tasked with implementing something as part of the user’s goal. This can be used to delegate a multi-step process that has its own agentic capabilities — deciding how best to implement the process and what tools to use.

## What does an agentic system actually do? The agentic flow: #

Agentic systems operate through three interconnected phases: Interaction (with users/events), Decision (via LLMs), and Action (tools, APIs, sub-agents).
  In Temporal, these are orchestrated together in a single durable workflow:

- Interactions : Agent responds to user prompts (e.g. chat) or external events (e.g. new data) and responds to the user.

- Decisions : Agent uses an LLM API to determine what actions to take. The agent may decide it requires additional user input.

- Actions : Agent executes specific activities, interfaces with external APIs, runs sub-agents, and uses knowledge bases as needed.

Here is a diagram of these three interactions working together as an agentic system, with Temporal orchestrating each: If you want to see an example of this system in action, check it out on GitHub: Temporal Agentic Workflow Example.

Together, these make a system that is dynamic, reacting to inputs from users, the LLM, and tool results.

## Agentic challenges #

So why isn’t the hype real yet? We have all of the ingredients — tools, LLMs, humans who would love assistance getting their tasks done. What’s missing?

Well, at Temporal, we’ve observed that there are some significant challenges to getting to production AI at scale.

- Humans can be unreliable, inaccurate, or non-responsive.

- Tool APIs and databases can go down.

- AI is inherently non-deterministic.

- Everything in this space is new (and changing constantly).

- Agentic systems are complex and hard to debug and test.

- Security is a problem that isn’t completely figured out yet.

Agentic systems often struggle to:

- Orchestrate complex multi-step interactions across distributed data stores and tools.

- Tolerate tool failure.

- Orchestrate multi-level processes.

- Hold state, potentially over long periods of time.

- Be durable: self-heal and retry until the LLM returns valid data.

- Have simple, generic implementations for human intervention such as approvals and input gathering.

- Provide insight into the agent’s performance.

- Tolerate human error and correction.

- Securely handle data and access on behalf of users.

- Ramp to production enterprise scale.

As we work with so many companies building AI systems, we hear these challenges every day.

The reasons the hype isn’t real can generally be summarized as: building complex distributed systems is hard and agentic systems are extremely complex distributed systems .

For more on this topic, check out this video on how AI agents are distributed systems in their own right.

## Agentic systems must be well-engineered distributed systems #

Agentic systems won’t work in production unless they’re well-engineered distributed systems.

That means they need to be stateful, fault-tolerant, observable, and able to coordinate both machines and humans over time. Which also means… they are best built with reliable orchestration.

Our position at Temporal is straightforward: If you want your agent to survive the real world, it needs Durable Execution. Durable Execution helps you easily solve the hard parts of distributed systems so you can get a reliable system by default, and you can focus on making something fun and useful for your users using this powerful technology.

That’s what we do and have done, and why we’re so invested in helping teams move from breakable demos to scalable, reliable systems.

## Building an agentic workflow framework #

Here is a simplified version of the agentic framework . Key elements of the Agentic Workflow:

- User interaction is enabled via Signals and self.add_message() — to send messages to the user

```

Signals

```

```

self.add_message()

```

- A Tool Planner activity which uses the LLM to plan tools

- Activities are used to execute Tools

- The interaction waits for user confirmation before running Tools

```

# Simplified version of https://github.com/temporal-community/temporal-ai-agent @ workflow . defn class AgentGoalWorkflow : def __init__( self ): … @ workflow . run async def run ( self ): while True : await workflow.wait_condition( # prompt OR confirm lambda : self .prompt_queue or ( self .waiting_for_confirm and self .confirm) ) # — handle prompt ——outcome: question/confirm/done— tool_data = await workflow.execute_activity( ToolActivities.agent_toolPlanner, { " prompt " : prompt, " history " : self .conversation_history}, ) self .add_message( " plan " , tool_data) # The planner thinks all arguments are ready and a tool should run # Ask the user to confirm if tool_data[ " next " ] == " confirm " : # agent is ready to run tool self .add_message( " agent " , tool_data[ " response " ]) elif tool_data[ " next " ] == " question " : # ask user for more info self .add_message( " agent " , tool_data[ " response " ]) elif tool_data[ " next " ] == " done " : # end chat return json.dumps( self .conversation_history) # — run tool if outcome is confirm and user confirms— elif self .waiting_for_confirm and self .confirm: result = await workflow.execute_activity(… self .add_message( " tool_result " , result) self .waiting_for_confirm = self .confirm = False # reset flags @ workflow.signal async def user_prompt( self , prompt: str ): … @ workflow.signal async def confirm( self ): self .confirm = True

```

```

# Simplified version of https://github.com/temporal-community/temporal-ai-agent @ workflow . defn class AgentGoalWorkflow : def __init__( self ): … @ workflow . run async def run ( self ): while True : await workflow.wait_condition( # prompt OR confirm lambda : self .prompt_queue or ( self .waiting_for_confirm and self .confirm) ) # — handle prompt ——outcome: question/confirm/done— tool_data = await workflow.execute_activity( ToolActivities.agent_toolPlanner, { " prompt " : prompt, " history " : self .conversation_history}, ) self .add_message( " plan " , tool_data) # The planner thinks all arguments are ready and a tool should run # Ask the user to confirm if tool_data[ " next " ] == " confirm " : # agent is ready to run tool self .add_message( " agent " , tool_data[ " response " ]) elif tool_data[ " next " ] == " question " : # ask user for more info self .add_message( " agent " , tool_data[ " response " ]) elif tool_data[ " next " ] == " done " : # end chat return json.dumps( self .conversation_history) # — run tool if outcome is confirm and user confirms— elif self .waiting_for_confirm and self .confirm: result = await workflow.execute_activity(… self .add_message( " tool_result " , result) self .waiting_for_confirm = self .confirm = False # reset flags @ workflow.signal async def user_prompt( self , prompt: str ): … @ workflow.signal async def confirm( self ): self .confirm = True

```

As a Temporal Worker, this application can be deployed just like any other Python application. Workers are stateless — all state is stored in the Temporal Service. Since they are stateless, any worker application crashes can be recovered from seamlessly, and I can scale this up to thousands of instances, each able to handle many many conversations concurrently.

What the agentic framework looks like

I can see the entire conversation and flow using the Temporal Workflow History, which I can also use for analyzing Agent success. We can add any Goals or Tools (even MCP tools!) to this framework, enabling the building of many different Agents. These goals and tool definitions are just passed as input to the workflow.

The workflow is dynamic, flowing and adapting to the conversation, with the agent planning tools based on user input: To try it for yourself, check out the Temporal Agent Framework and explore. Keep in mind, since it’s just a Temporal application, it will scale and be durable as you need it to. I’m excited about this, since it delivers on the needs of agentic applications mentioned above. Plus, it’s a lot of fun to work with.

## Production agentic AI at scale — powered by Temporal #

Fortunately, Temporal was designed to solve the pain of building complex distributed systems. Durable Execution with Temporal makes orchestration of agentic processes simple.

Agents implemented as Workflows can run for as long as you need them to. Failures are easy to retry, human interaction is simple and flexible, and scale is trivial. It really is that simple.

Want to see what I mean? Check out this webinar , in which we walk through our agentic AI framework , how we built it, and we try to make it break — but Temporal keeps it bulletproof. I’ve talked to so many people who’ve taken the framework for a spin and built cool agents with it — some are in production now. Building with this framework is fun — I hope you enjoy working with it as so many others have.

If you’re like me, then I’m sure you want to dive deeper into how Temporal makes systems of all kinds durable, scalable, flexible, flexible, and simple to build. If so, you can check out these resources:

- Why Temporal Replaces Traditional State Machines for Distributed Applications

- Building Reliable Distributed Systems with Temporal: Error Handling & Workflow Management (on the AWS Developers Youtube channel)

- A Better Way To Build Modern Applications

- Events are the wrong abstraction

If you have questions and want to learn more, feel free to reach out:

- Join the Temporal Community Slack (channel #topic-ai ). I’m there and always willing to chat!

- Talk to an expert and discuss your specific use case.

- Take us for a whirl for yourself: sign up for Temporal Cloud and get $1,000 in free credits.

Temporal Cloud

Ready to see for yourself?

Sign up for Temporal Cloud today and get $1,000 in free credits.

### More Posts

- Dec 10, 2025 Temporal Voices 5 MIN READ The city of systems: Temporal, Kafka, and Nexus

The city of systems: Temporal, Kafka, and Nexus

- Nov 12, 2025 Temporal Voices 10 MIN READ Of course you can build dynamic AI agents with Temporal

Of course you can build dynamic AI agents with Temporal

Build invincible applications

It sounds like magic, we promise it's not.

ALL SYSTEMS OPERATIONAL

Discover

Explore

Developers

Community

Company

2025 © Temporal Technologies. All Rights Reserved.