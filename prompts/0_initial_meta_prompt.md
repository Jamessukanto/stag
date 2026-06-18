Your task is NOT to implement the project.

Your task is NOT to design the software architecture directly.

Your task is to design a prompting strategy for Cursor.

Think of Cursor as a team of specialized engineers, each with its own chat session.

The goal is to preserve architecture and avoid monolithic AI-generated code.

Assumptions:

* I primarily use Python.
* I use Cursor Agent.
* Each new Cursor chat begins in Plan Mode.
* I prefer separate chats for separate subsystems.
* Implementation should follow TDD.
* Long-term maintainability is more important than minimizing the number of chats.
* A lightweight React frontend may be added later.
* I want clean boundaries between modules.

Given the project brief (provided below), produce a prompt suite.

The primary output should be a sequence of Cursor chats.

For each chat, provide:

# Chat name

Examples:

* Architecture
* Data
* Models
* Evaluation
* Experiments
* API
* Frontend
* Infrastructure

# Purpose

One or two sentences describing what this chat owns.

# Dependencies

What previous chats or artifacts this chat may rely on.

# Initial Plan Mode prompt

This is the most important part.

Generate the exact prompt I should paste into a fresh Cursor chat while Plan Mode is enabled.

The prompt should:

* constrain scope;
* define responsibilities;
* specify allowed dependencies;
* specify forbidden dependencies;
* preserve separation of concerns;
* mention testing expectations when relevant;
* avoid implementation details unless appropriate.

Write these prompts as ready-to-paste text.

# Follow-up implementation prompt

After I approve the plan, generate the exact prompt I should use to transition from planning to implementation.

These prompts should:

* enforce TDD;
* generate tests first;
* verify failures before implementation;
* run pytest after changes;
* avoid modifying tests to satisfy implementations;
* preserve interfaces and architectural boundaries.

# Why this chat exists

Explain briefly why this subsystem deserves its own chat rather than sharing context with others.

# Ordering

Recommend the order in which chats should be executed.

Indicate which chats can be parallelized.

The output should focus almost entirely on producing prompts for Cursor chats.

Do not produce architecture documents, milestone documents, risk documents, or global rule documents as standalone outputs.

Instead, incorporate architectural constraints, risks, maintainability concerns, and testing philosophy into the prompts themselves.

Optimize for:

* modularity;
* architecture preservation;
* reproducibility;
* minimal cross-chat coupling;
* future extensibility;
* preventing AI agents from turning the repository into a monolith.

I will provide the project brief below:
