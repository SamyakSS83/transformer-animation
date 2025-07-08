from manim import *

class Tokenization(Scene):
    def construct(self):
        title = Text("Step 1: Tokenization", font_size=40).to_edge(UP)
        sentence = Text("\"I am a human being\"", font_size=36).next_to(title, DOWN, buff=1)
        
        tokens = VGroup(*[
            Text(word, font_size=36).set_color(BLUE) 
            for word in ["I", "am", "a", "human", "being"]
        ]).arrange(RIGHT, buff=0.8).next_to(sentence, DOWN, buff=2)
        
        arrows = VGroup(*[
            Arrow(sentence.get_bottom(), token.get_top(), buff=0.2, color=YELLOW)
            for token in tokens
        ])
        
        self.play(Write(title))
        self.play(Write(sentence))
        self.play(
            LaggedStart(*[Create(arrow) for arrow in arrows], lag_ratio=0.3),
            LaggedStart(*[Write(token) for token in tokens], lag_ratio=0.3),
            run_time=2
        )
        self.wait(2)

class Embedding(Scene):
    def construct(self):
        title = Text("Step 2: Embedding + Positional Encoding", font_size=40).to_edge(UP)
        tokens = VGroup(*[
            Text(word, font_size=24) 
            for word in ["I", "am", "a", "human", "being"]
        ]).arrange(RIGHT, buff=0.5).next_to(title, DOWN, buff=1)
        
        # Embedding matrices
        embed_matrices = VGroup(*[
            Matrix([[f"E_{i+1}{j+1}" for j in range(4)], 
                   [":"]], 
                  h_buff=0.8, v_buff=0.5)
            .scale(0.5)
            for i in range(5)
        ]).arrange(RIGHT, buff=0.3).next_to(tokens, DOWN, buff=1.5)
        
        # Positional encoding
        pos_encodings = VGroup(*[
            Matrix([[f"P_{i+1}{j+1}" for j in range(4)], 
                   [":"]], 
                  h_buff=0.8, v_buff=0.5)
            .scale(0.5)
            .set_color(GREEN)
            for i in range(5)
        ]).arrange(RIGHT, buff=0.3).next_to(embed_matrices, DOWN, buff=1)
        
        # Final embeddings
        final_embeddings = VGroup(*[
            Matrix([[f"X_{i+1}{j+1}" for j in range(4)], 
                   [":"]], 
                  h_buff=0.8, v_buff=0.5)
            .scale(0.5)
            .set_color(YELLOW)
            for i in range(5)
        ]).arrange(RIGHT, buff=0.3).next_to(pos_encodings, DOWN, buff=1)
        
        self.play(Write(title))
        self.play(Write(tokens))
        self.play(
            LaggedStart(*[Write(m) for m in embed_matrices], lag_ratio=0.2),
            run_time=2
        )
        self.play(
            LaggedStart(*[Write(p) for p in pos_encodings], lag_ratio=0.2),
            run_time=2
        )
        self.play(
            LaggedStart(*[Write(f) for f in final_embeddings], lag_ratio=0.2),
            run_time=2
        )
        self.wait(2)

class AttentionHead(Scene):
    def construct(self):
        title = Text("Step 3: Self-Attention Mechanism", font_size=40).to_edge(UP)
        input_matrix = Matrix([["X_11", "..", "X_1d"],
                             [":", ".", ":"],
                             ["X_n1", "..", "X_nd"]], 
                            h_buff=1.2, v_buff=0.8).scale(0.6).to_edge(LEFT)
        
        # Attention components
        weights = VGroup(
            Matrix([["W_Q"]], h_buff=1.2).set_color(BLUE),
            Matrix([["W_K"]], h_buff=1.2).set_color(RED),
            Matrix([["W_V"]], h_buff=1.2).set_color(GREEN)
        ).arrange(DOWN, buff=1).next_to(input_matrix, RIGHT, buff=2)
        
        outputs = VGroup(
            Matrix([["Q_1", "..", "Q_d"]], h_buff=1.2).set_color(BLUE),
            Matrix([["K_1", "..", "K_d"]], h_buff=1.2).set_color(RED),
            Matrix([["V_1", "..", "V_d"]], h_buff=1.2).set_color(GREEN)
        ).arrange(DOWN, buff=1).to_edge(RIGHT)
        
        self.play(Write(title))
        self.play(Write(input_matrix))
        self.play(LaggedStart(*[Write(w) for w in weights], lag_ratio=0.3))
        self.play(LaggedStart(*[TransformFromCopy(w, o) for w, o in zip(weights, outputs)], lag_ratio=0.3))
        
        # Attention calculation
        attn_eq = MathTex(r"\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V"
                        ).next_to(outputs, DOWN, buff=1)
        self.play(Write(attn_eq))
        self.wait(2)

class MultiHeadAttention(Scene):
    def construct(self):
        title = Text("Step 4: Multi-Head Attention", font_size=40).to_edge(UP)
        
        # Create multiple attention heads
        heads = VGroup(*[
            VGroup(
                Rectangle(height=2, width=3, fill_opacity=0.2, color=color),
                Text(f"Head {i+1}", font_size=24)
            )
            for i, color in enumerate([BLUE, GREEN, RED, YELLOW])
        ]).arrange_in_grid(2, 2, buff=1).scale(0.8).next_to(title, DOWN, buff=1)
        
        # Concatenation and projection
        concat_matrix = Matrix([["H1_1", "..", "H8_1"],
                              [":", ".", ":"],
                              ["H1_n", "..", "H8_n"]], 
                             h_buff=1.2).next_to(heads, DOWN, buff=1.5)
        proj_matrix = Matrix([["O_1"], [":"], ["O_d"]]).next_to(concat_matrix, RIGHT, buff=1.5)
        
        self.play(Write(title))
        self.play(LaggedStart(*[Create(h) for h in heads], lag_ratio=0.3))
        self.play(Write(concat_matrix))
        self.play(Write(proj_matrix))
        
        arrows = VGroup(
            Arrow(heads.get_bottom(), concat_matrix.get_top(), buff=0.2),
            Arrow(concat_matrix.get_right(), proj_matrix.get_left(), buff=0.2)
        )
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows]))
        self.wait(2)

class FeedForward(Scene):
    def construct(self):
        title = Text("Step 5: Feed-Forward Network", font_size=40).to_edge(UP)
        input_matrix = Matrix([["O_1"], [":"], ["O_d"]]).next_to(title, DOWN, buff=1)
        
        ffn = VGroup(
            Rectangle(height=3, width=4, color=BLUE, fill_opacity=0.2),
            Text("FFN\n(ReLU → Linear)", font_size=30)
        ).next_to(input_matrix, DOWN, buff=1.5)
        
        output_matrix = Matrix([["F_1"], [":"], ["F_d"]]).next_to(ffn, DOWN, buff=1.5)
        
        self.play(Write(title))
        self.play(Write(input_matrix))
        self.play(Create(ffn))
        self.play(Write(output_matrix))
        self.play(
            GrowArrow(Arrow(input_matrix.get_bottom(), ffn.get_top(), buff=0.2)),
            GrowArrow(Arrow(ffn.get_bottom(), output_matrix.get_top(), buff=0.2))
        )
        self.wait(2)

class EncoderBlock(Scene):
    def construct(self):
        title = Text("Encoder Block", font_size=40).to_edge(UP)
        
        components = VGroup(
            Text("Input Embedding", font_size=24),
            Text("Multi-Head\nAttention", font_size=24),
            Text("Add & Norm", font_size=24),
            Text("Feed Forward", font_size=24),
            Text("Add & Norm", font_size=24),
            Text("Output", font_size=24)
        ).arrange(RIGHT, buff=1.5).scale(0.8).next_to(title, DOWN, buff=1)
        
        arrows = VGroup(*[
            Arrow(start.get_right(), end.get_left(), buff=0.2)
            for start, end in zip(components[:-1], components[1:])
        ])
        
        residuals = VGroup(
            CurvedArrow(components[0].get_top(), components[2].get_top(), angle=-0.5),
            CurvedArrow(components[2].get_top(), components[4].get_top(), angle=-0.5)
        ).set_color(YELLOW)
        
        self.play(Write(title))
        self.play(LaggedStart(*[Write(c) for c in components], lag_ratio=0.3))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.3))
        self.play(LaggedStart(*[Create(r) for r in residuals], lag_ratio=0.3))
        self.wait(2)
