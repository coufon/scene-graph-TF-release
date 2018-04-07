# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import server_pb2 as server__pb2


class WorkerStub(object):

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.RunTask = channel.unary_unary(
        '/proto_server.Worker/RunTask',
        request_serializer=server__pb2.Task.SerializeToString,
        response_deserializer=server__pb2.Result.FromString,
        )


class WorkerServicer(object):

  def RunTask(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_WorkerServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'RunTask': grpc.unary_unary_rpc_method_handler(
          servicer.RunTask,
          request_deserializer=server__pb2.Task.FromString,
          response_serializer=server__pb2.Result.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'proto_server.Worker', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
