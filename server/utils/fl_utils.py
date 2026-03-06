from flwr.server.client_manager import SimpleClientManager  
from flwr.server.app import start_grpc_server

def call_get_parameters(cp, ins):

    return cp.get_parameters(ins, timeout=None)


def call_fit(cp, fit_ins):

    return cp.fit(fit_ins, timeout=None)


def start_flower_grpc_server(server_address: str):

    client_manager = SimpleClientManager()

    grpc_server = start_grpc_server(
        server_address=server_address,
        client_manager=client_manager,
        max_message_length=536870912,
    )

    return grpc_server, client_manager