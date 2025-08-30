# PromptGuard

Welcome to PromptGuard.
A free, open-source prompt hardening and injection detection library.

[![MIT license](https://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)

This library is currently a work in progress, but the goal is to deliver the most complete open source solution to harden GenAI applications in this AI era.

## Isn't there plenty of tools out there?

There's plenty of similar tools, but none quite hits the spot. Furthermore, most tools are behind paywall.
PromptGuard is thought to be easy to use, light, and completely modular.
It will address one problem and it will do it well: prompt injections.

Prompt injections are the main vulnerability that arose from large language models, and AI pentesting is mainly focused on this at the moment.
Essentially, the key to prompt injection lies into altering a model's behaviour, by for example making it output slurs, or execute instructions that contrast with its original system design prompt.
Most applications simply use large language models as chatbots at the moment, without granting access to any sensitive data. In this scenario, the worst a model can do is output words or phrases that are against its original intended purpose.
While this might sound like a big deal, in reality it does not pose much of a threat to a company, except in very particular cases of scandal.

There are, however, increasing situations in which models are deeply interconnected into a company's infrastructure:
Retrieval-Augmented-Generation (RAG) models, MCP Servers, models that are fine-tuned on private company data to offer detailed support. In all these cases, the risk for customerìs personal data exfiltration, company documents leakage and similar events is indeed an existing threat.

This is what PromptGuard is trying to do: being a complete utility implementing full modular pipelines to defend against prompt injections, while still being completely free and open source. No paywall, no black boxes.

## Installation Guide

Still WIP - First all the main modules need to be completed!
