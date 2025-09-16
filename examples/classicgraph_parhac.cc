// Copyright 2025
//
// Reads a ClassicGraph saved by our Rust project (parlayANN-like format),
// treats it as an undirected graph with unit weights, runs ParHAC, and
// writes the resulting dendrogram to a CSV file.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <cstring>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/hac/parhac.h"
#include "in_memory/status_macros.h"

using graph_mining::in_memory::ClustererConfig;
using graph_mining::in_memory::Dendrogram;
using graph_mining::in_memory::DendrogramNode;
using graph_mining::in_memory::ParHacClusterer;
using graph_mining::in_memory::SimpleUndirectedGraph;
using graph_mining::in_memory::CopyGraph;

namespace {

absl::Status ReadClassicGraphAutoWeights(
    const std::string& path, SimpleUndirectedGraph* g) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return absl::NotFoundError("Failed to open input file: " + path);
  }

  auto read_u32 = [&](uint32_t& v) -> bool {
    char buf[4];
    if (!in.read(buf, 4)) return false;
    v = static_cast<uint32_t>(static_cast<unsigned char>(buf[0])) |
        (static_cast<uint32_t>(static_cast<unsigned char>(buf[1])) << 8) |
        (static_cast<uint32_t>(static_cast<unsigned char>(buf[2])) << 16) |
        (static_cast<uint32_t>(static_cast<unsigned char>(buf[3])) << 24);
    return true;
  };

  uint32_t n = 0;
  uint32_t r = 0;
  if (!read_u32(n) || !read_u32(r)) {
    return absl::InvalidArgumentError("Invalid ClassicGraph header");
  }

  std::vector<uint32_t> degrees(n);
  for (uint32_t i = 0; i < n; ++i) {
    if (!read_u32(degrees[i])) {
      return absl::InvalidArgumentError("Failed reading degrees");
    }
  }

  // Compute number of edges to detect whether weights are present.
  uint64_t total_edges = 0;
  for (uint32_t i = 0; i < n; ++i) total_edges += degrees[i];

  // Determine file size to see if weights are appended.
  std::streampos pos_after_header_and_degrees = in.tellg();
  in.seekg(0, std::ios::end);
  std::streampos file_size = in.tellg();
  in.seekg(pos_after_header_and_degrees);

  const uint64_t bytes_for_edges = total_edges * 4ULL;
  const uint64_t bytes_for_weights = total_edges * 4ULL;
  const uint64_t remaining_bytes = static_cast<uint64_t>(file_size - pos_after_header_and_degrees);

  const bool has_weights = remaining_bytes >= (bytes_for_edges + bytes_for_weights);

  if (!has_weights) {
    // Edges are written per node in order; read sequentially with unit weights.
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t deg = degrees[i];
      for (uint32_t j = 0; j < deg; ++j) {
        uint32_t nbr = 0;
        if (!read_u32(nbr)) {
          return absl::InvalidArgumentError("Failed reading neighbor list");
        }
        RETURN_IF_ERROR(g->AddEdge(static_cast<int64_t>(i),
                                   static_cast<int64_t>(nbr), 1.0));
      }
    }
    return absl::OkStatus();
  }

  // Weighted: read all edges, then all weights, then add edges with weights in order.
  std::vector<uint32_t> edges;
  edges.resize(static_cast<size_t>(total_edges));
  // Read all edge indices
  for (uint64_t e = 0; e < total_edges; ++e) {
    if (!read_u32(edges[static_cast<size_t>(e)])) {
      return absl::InvalidArgumentError("Failed reading edges block");
    }
  }

  // Read all weights as raw little-endian u32 then bit-cast to float
  std::vector<float> weights;
  weights.resize(static_cast<size_t>(total_edges));
  for (uint64_t e = 0; e < total_edges; ++e) {
    uint32_t bits = 0;
    if (!read_u32(bits)) {
      return absl::InvalidArgumentError("Failed reading weights block");
    }
    float f;
    static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
    std::memcpy(&f, &bits, sizeof(float));
    weights[static_cast<size_t>(e)] = f;
  }

  // Distribute edges and weights per node in the same traversal order.
  uint64_t offset = 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t deg = degrees[i];
    for (uint32_t j = 0; j < deg; ++j, ++offset) {
      uint32_t nbr = edges[static_cast<size_t>(offset)];
      double w = static_cast<double>(weights[static_cast<size_t>(offset)]);
      RETURN_IF_ERROR(g->AddEdge(static_cast<int64_t>(i),
                                 static_cast<int64_t>(nbr), w));
    }
  }

  return absl::OkStatus();
}

absl::Status WriteDendrogramCsv(const Dendrogram& dendro,
                                const std::string& out_path) {
  std::ofstream out(out_path);
  if (!out) {
    return absl::InvalidArgumentError("Failed to open output file: " + out_path);
  }
  out << "id,parent_id,merge_similarity\n";
  const auto& nodes = dendro.Nodes();
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto parent_id = nodes[i].parent_id == Dendrogram::kNoParentId
                         ? -1
                         : static_cast<long long>(nodes[i].parent_id);
    out << i << "," << parent_id << "," << nodes[i].merge_similarity << "\n";
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr
        << "Usage: classicgraph_parhac <input_classicgraph.bin> <output_dendrogram.csv> [weight_threshold] [epsilon]\n";
    return 1;
  }

  const std::string input_path = argv[1];
  const std::string output_path = argv[2];
  double weight_threshold = 0.0;
  double epsilon = 0.1;
  if (argc >= 4) {
    weight_threshold = std::stod(argv[3]);
  }
  if (argc >= 5) {
    epsilon = std::stod(argv[4]);
  }

  SimpleUndirectedGraph input_graph;
  absl::Status read_st = ReadClassicGraphAutoWeights(input_path, &input_graph);
  if (!read_st.ok()) {
    std::cerr << "Error reading ClassicGraph: " << read_st << "\n";
    return 1;
  }

  ParHacClusterer clusterer;
  absl::Status st = CopyGraph(input_graph, clusterer.MutableGraph());
  if (!st.ok()) {
    std::cerr << "Error importing graph: " << st << "\n";
    return 1;
  }

  ClustererConfig config;
  config.mutable_parhac_clusterer_config()->set_weight_threshold(
      weight_threshold);
  config.mutable_parhac_clusterer_config()->set_epsilon(epsilon);

  auto dendro_or = clusterer.HierarchicalCluster(config);
  if (!dendro_or.ok()) {
    std::cerr << "Error clustering: " << dendro_or.status() << "\n";
    return 1;
  }
  Dendrogram dendro = std::move(dendro_or).value();

  st = WriteDendrogramCsv(dendro, output_path);
  if (!st.ok()) {
    std::cerr << "Error writing dendrogram: " << st << "\n";
    return 1;
  }

  return 0;
}