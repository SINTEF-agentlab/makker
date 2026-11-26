"""The MCP server: a Makker session's tools, for another agent to call.

``makker mcp`` serves one session over stdin/stdout to an MCP client. The client's
own agent decides what to run; Makker runs it in the session's Julia, records it
in the trace, and serves the figures. ``server`` is the protocol side, ``runtime``
the session and how its tools are called, ``specs`` the tool list, ``pages`` the
live-figure links, and ``cli`` the command.

Importing this package must stay cheap: a client gives a server seconds to
answer, so everything that pulls in the agent stack is imported after that.
"""
