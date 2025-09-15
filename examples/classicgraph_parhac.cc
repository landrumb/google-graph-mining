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

absl::StatusOr<SimpleUndirectedGraph> ReadClassicGraphAsUndirectedUnitWeights(
    const std::string& path) {
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

  SimpleUndirectedGraph g;
  // Edges are written per node in order; read sequentially.
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t deg = degrees[i];
    for (uint32_t j = 0; j < deg; ++j) {
      uint32_t nbr = 0;
      if (!read_u32(nbr)) {
        return absl::InvalidArgumentError("Failed reading neighbor list");
      }
      // Unit weight and undirected: SimpleUndirectedGraph ensures symmetry and
      // avoids duplicate parallel edges via map overwrite.
      RETURN_IF_ERROR(g.AddEdge(static_cast<int64_t>(i),
                                static_cast<int64_t>(nbr), 1.0));
    }
  }

  return g;
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

  auto g_or = ReadClassicGraphAsUndirectedUnitWeights(input_path);
  if (!g_or.ok()) {
    std::cerr << "Error reading ClassicGraph: " << g_or.status() << "\n";
    return 1;
  }
  SimpleUndirectedGraph input_graph = std::move(g_or).value();

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


