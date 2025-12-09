//
// NeuroPatternSynth.swift
// Neural Cellular Automata pattern generator + mutation evolution
// Inspired by modern research on self-organizing systems
//
// Swift 5+
//

import Foundation
import Accelerate

// =========================================================
// MARK: - Utility Structures
// =========================================================

struct Vector {
    var values: [Double]
    
    static func +(lhs: Vector, rhs: Vector) -> Vector {
        Vector(values: zip(lhs.values, rhs.values).map(+))
    }
    static func *(lhs: Double, rhs: Vector) -> Vector {
        Vector(values: rhs.values.map { lhs * $0 })
    }
}

func sigmoid(_ x: Double) -> Double { 1 / (1 + exp(-x)) }
func relu(_ x: Double) -> Double { max(0, x) }
func clamp(_ x: Double, min mn: Double, max mx: Double) -> Double {
    return x < mn ? mn : (x > mx ? mx : x)
}

// =========================================================
// MARK: - NCA Cell Neural Model
// =========================================================

struct NCACell {
    // State channels: e.g., R, G, B, Aux
    var state: [Double] // length >= 4
    
    init() {
        self.state = [
            Double.random(in: 0...1),
            Double.random(in: 0...1),
            Double.random(in: 0...1),
            Double.random(in: 0...1)
        ]
    }
}

// A neural rule: small dense layer network transforming neighborhood state
class NCARule {
    var W1: [[Double]]  // layer 1 weights
    var b1: [Double]
    var W2: [[Double]]  // layer 2 weights
    var b2: [Double]
    
    init(input: Int, hidden: Int, output: Int) {
        func rand(_ m:Int, _ n:Int) -> [[Double]] {
            (0..<m).map { _ in (0..<n).map { _ in Double.random(in: -0.2...0.2) }}
        }
        self.W1 = rand(hidden, input)
        self.b1 = (0..<hidden).map { _ in Double.random(in: -0.1...0.1) }
        self.W2 = rand(output, hidden)
        self.b2 = (0..<output).map { _ in Double.random(in: -0.1...0.1) }
    }
    
    func forward(_ x: [Double]) -> [Double] {
        // h = relu(W1x + b1)
        var h = [Double](repeating: 0, count: b1.count)
        for i in 0..<W1.count {
            var s = b1[i]
            for j in 0..<x.count { s += W1[i][j] * x[j] }
            h[i] = relu(s)
        }
        // out = sigmoid(W2h + b2)
        var out = [Double](repeating: 0, count: b2.count)
        for i in 0..<W2.count {
            var s = b2[i]
            for j in 0..<h.count { s += W2[i][j] * h[j] }
            out[i] = sigmoid(s)
        }
        return out
    }
    
    func mutate(intensity: Double = 0.02) {
        for i in 0..<W1.count {
            for j in 0..<W1[i].count {
                W1[i][j] += Double.random(in: -intensity...intensity)
            }
        }
        for i in 0..<b1.count { b1[i] += Double.random(in: -intensity...intensity) }
        
        for i in 0..<W2.count {
            for j in 0..<W2[i].count {
                W2[i][j] += Double.random(in: -intensity...intensity)
            }
        }
        for i in 0..<b2.count { b2[i] += Double.random(in: -intensity...intensity) }
    }
}

// =========================================================
// MARK: - NCA Grid
// =========================================================

class NCAGrid {
    let width: Int
    let height: Int
    var grid: [[NCACell]]
    let rule: NCARule
    
    init(w: Int, h: Int, rule: NCARule) {
        self.width = w
        self.height = h
        self.rule = rule
        self.grid = (0..<h).map { _ in (0..<w).map { _ in NCACell() } }
    }
    
    func getNeighborhood(x: Int, y: Int) -> [Double] {
        var vals: [Double] = []
        for dy in -1...1 {
            for dx in -1...1 {
                let nx = (x + dx + width) % width
                let ny = (y + dy + height) % height
                vals.append(contentsOf: grid[ny][nx].state)
            }
        }
        return vals
    }
    
    func step() {
        var newState = grid
        for y in 0..<height {
            for x in 0..<width {
                let neigh = getNeighborhood(x: x, y: y)
                let out = rule.forward(neigh)
                for c in 0..<newState[y][x].state.count {
                    let updated = (grid[y][x].state[c] * 0.6 + out[c] * 0.4)
                    newState[y][x].state[c] = clamp(updated, min: 0, max: 1)
                }
            }
        }
        grid = newState
    }
    
    func asciiRender(colorChannel: Int = 0) -> String {
        var s = ""
        for y in 0..<height {
            for x in 0..<width {
                let v = grid[y][x].state[colorChannel]
                let chars = " .:-=+*#%@"
                let idx = Int(v * Double(chars.count-1))
                s.append(chars[chars.index(chars.startIndex, offsetBy: idx)])
            }
            s.append("\n")
        }
        return s
    }
    
    func avgVariance() -> Double {
        let flat = grid.flatMap { $0 }.flatMap { $0.state }
        let mean = flat.reduce(0,+) / Double(flat.count)
        let varSum = flat.map { pow($0 - mean,2) }.reduce(0,+)
        return varSum / Double(flat.count)
    }
}

// =========================================================
// MARK: - Evolution Driver
// =========================================================

func evolvePatterns() {
    print("=== NeuroPatternSynth Demo ===")
    
    let parentRule = NCARule(input: 9*4, hidden: 24, output: 4)
    let parent = NCAGrid(w: 32, h: 16, rule: parentRule)
    
    for gen in 0..<8 {
        print("\n--- Generation \(gen) ---")
        
        // Run steps
        for _ in 0..<30 { parent.step() }
        
        // Render
        print(parent.asciiRender(colorChannel: Int.random(in: 0..<4)))
        
        // Score = variance of states (want structured complexity)
        let score = parent.avgVariance()
        print("Score:", String(format: "%.4f", score))
        
        // Mutate rule slightly for next generation
        parent.rule.mutate(intensity: 0.01)
    }
}

evolvePatterns()
