from manim import *
from manim import config
import numpy as np

class TransformerOverviewScene(Scene):
    def construct(self):
        # Title
        title = Text("The Transformer Architecture", font_size=48, color=BLUE)
        subtitle = Text("Attention is All You Need", font_size=32, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        
        self.play(Write(title))
        self.wait(0.5)
        self.play(Write(subtitle))
        self.wait(2)
        self.play(title_group.animate.to_edge(UP))
        
        # Create the main transformer diagram
        self.create_transformer_diagram()
        
    def create_transformer_diagram(self):
        # Input tokens
        input_text = Text("I love machine learning", font_size=24)
        tokens = ["I", "love", "machine", "learning"]
        
        token_boxes = []
        for token in tokens:
            box = Rectangle(width=1.5, height=0.8, color=BLUE, fill_opacity=0.3)
            text = Text(token, font_size=20)
            token_group = VGroup(box, text)
            token_boxes.append(token_group)
        
        input_row = VGroup(*token_boxes).arrange(RIGHT, buff=0.2).shift(DOWN * 3)
        
        # Show tokenization
        self.play(Write(input_text.move_to(DOWN * 3)))
        self.wait(1)
        self.play(Transform(input_text, input_row))
        self.wait(1)
        
        # Embedding layer - create matrices instead of vectors
        embed_label = Text("Embedding Layer", font_size=20, color=GREEN)
        embed_matrices = []
        for i in range(4):
            # Create a matrix representation with multiple rows
            matrix_rows = []
            for row in range(6):  # 6 dimensions
                rect = Rectangle(width=0.8, height=0.15, color=GREEN, fill_opacity=0.6)
                matrix_rows.append(rect)
            matrix = VGroup(*matrix_rows).arrange(DOWN, buff=0.02)
            embed_matrices.append(matrix)
        
        embed_row = VGroup(*embed_matrices).arrange(RIGHT, buff=0.5)
        embed_group = VGroup(embed_label, embed_row).arrange(DOWN, buff=0.3).move_to(DOWN * 1.5)
        
        self.play(FadeIn(embed_group))
        self.wait(1)
        
        # Positional encoding - also matrices
        pos_label = Text("+ Positional Encoding", font_size=18, color=ORANGE)
        pos_matrices = []
        for i in range(4):
            matrix_rows = []
            for row in range(6):  # Same dimensions
                rect = Rectangle(width=0.8, height=0.15, color=ORANGE, fill_opacity=0.4)
                matrix_rows.append(rect)
            matrix = VGroup(*matrix_rows).arrange(DOWN, buff=0.02)
            pos_matrices.append(matrix)
        
        pos_row = VGroup(*pos_matrices).arrange(RIGHT, buff=0.5)
        pos_group = VGroup(pos_label, pos_row).arrange(DOWN, buff=0.3).move_to(UP * 0.2)
        
        self.play(FadeIn(pos_group))
        
        # Show addition - position plus signs better
        plus_signs = []
        for i in range(4):
            plus = Text("+", font_size=20, color=WHITE)
            plus.move_to((embed_matrices[i].get_center() + pos_matrices[i].get_center()) / 2)
            plus_signs.append(plus)
        
        self.play(*[Write(plus) for plus in plus_signs])
        self.wait(1)
        
        # Transformer block
        self.create_attention_mechanism()
        
    def create_attention_mechanism(self):
        # Clear previous content
        self.clear()
        
        # Title for attention
        title = Text("Self-Attention Mechanism", font_size=36, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))
        
        # Input sequence
        words = ["The", "cat", "sat", "on", "mat"]
        input_boxes = []
        
        for word in words:
            box = Rectangle(width=1.2, height=0.6, color=BLUE, fill_opacity=0.3)
            text = Text(word, font_size=16)
            word_group = VGroup(box, text)
            input_boxes.append(word_group)
        
        input_sequence = VGroup(*input_boxes).arrange(RIGHT, buff=0.1).shift(UP * 2)
        self.play(FadeIn(input_sequence))
        self.wait(1)
        
        # Q, K, V transformation - show as matrices
        qkv_labels = [
            Text("Query (Q) Matrix", font_size=18, color=RED),
            Text("Key (K) Matrix", font_size=18, color=GREEN),
            Text("Value (V) Matrix", font_size=18, color=YELLOW)
        ]
        
        qkv_matrices = []
        for i, color in enumerate([RED, GREEN, YELLOW]):
            # Create actual matrix representation
            matrix_elements = []
            for row in range(5):  # 5 tokens
                row_elements = []
                for col in range(4):  # 4 dimensions
                    element = Rectangle(width=0.3, height=0.2, color=color, fill_opacity=0.6)
                    row_elements.append(element)
                matrix_row = VGroup(*row_elements).arrange(RIGHT, buff=0.02)
                matrix_elements.append(matrix_row)
            matrix = VGroup(*matrix_elements).arrange(DOWN, buff=0.02)
            qkv_matrices.append(matrix)
        
        qkv_groups = []
        for i in range(3):
            group = VGroup(qkv_labels[i], qkv_matrices[i]).arrange(DOWN, buff=0.3)
            qkv_groups.append(group)
        
        qkv_row = VGroup(*qkv_groups).arrange(RIGHT, buff=1).shift(DOWN * 0.5)
        
        # Show transformations with colored arrows
        for i, group in enumerate(qkv_groups):
            arrows = []
            for j in range(5):
                arrow = Arrow(
                    start=input_boxes[j].get_bottom(),
                    end=qkv_matrices[i].get_top(),
                    color=qkv_labels[i].color,
                    buff=0.1
                )
                arrows.append(arrow)
            
            self.play(FadeIn(group))
            self.play(*[GrowArrow(arrow) for arrow in arrows])
            self.wait(0.5)
        
        self.wait(2)
        
        # Show attention calculation
        self.create_attention_calculation()
        
    def create_attention_calculation(self):
        # Clear and show attention formula
        self.clear()
        
        title = Text("Attention Calculation", font_size=36, color=BLUE)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        # Main attention formula with proper mathematical notation
        formula = MathTex(
            r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            font_size=28
        )
        formula.move_to(UP * 1)
        
        self.play(Write(formula))
        self.wait(1)
        
        # Step by step breakdown with mathematical notation
        step1 = MathTex(r"\text{Step 1: Compute } QK^T \text{ (similarity scores)}", font_size=20)
        step2 = MathTex(r"\text{Step 2: Scale by } \sqrt{d_k}", font_size=20)
        step3 = Text("Step 3: Apply softmax (attention weights)", font_size=20)
        step4 = Text("Step 4: Multiply by V (weighted values)", font_size=20)
        
        steps = VGroup(step1, step2, step3, step4).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        steps.shift(DOWN * 1)
        
        for step in steps:
            self.play(Write(step))
            self.wait(0.8)
        
        self.wait(2)
        
        # Show attention matrix visualization
        self.create_attention_matrix()
        
    def create_attention_matrix(self):
        # Clear and create attention matrix visualization
        self.clear()
        
        title = Text("Attention Weights Visualization", font_size=32, color=BLUE)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        # Words with more realistic attention relationships
        words = ["The", "cat", "sat", "on", "mat"]
        
        # Create more realistic attention matrix reflecting linguistic relationships
        matrix_size = 5
        attention_matrix = np.array([
            [0.05, 0.1, 0.2, 0.05, 0.6],   # "The" -> strongly attends to "mat" (determiner-noun)
            [0.1, 0.2, 0.6, 0.05, 0.05],   # "cat" -> strongly attends to "sat" (subject-verb)  
            [0.05, 0.7, 0.1, 0.1, 0.05],   # "sat" -> strongly attends to "cat" (verb-subject)
            [0.05, 0.1, 0.1, 0.1, 0.65],   # "on" -> strongly attends to "mat" (preposition-object)
            [0.8, 0.05, 0.05, 0.05, 0.05]  # "mat" -> strongly attends to "The" (noun-determiner)
        ])
        
        # Create visual matrix with better formatting
        squares = VGroup()
        for i in range(matrix_size):
            row = VGroup()
            for j in range(matrix_size):
                intensity = attention_matrix[i][j]
                color = interpolate_color(WHITE, RED, intensity)
                square = Square(side_length=0.8, color=color, fill_opacity=0.9, stroke_color=BLACK, stroke_width=1)
                value_text = Text(f"{intensity:.2f}", font_size=14, color=BLACK if intensity < 0.5 else WHITE)
                square_group = VGroup(square, value_text)
                row.add(square_group)
            row.arrange(RIGHT, buff=0)
            squares.add(row)
        
        squares.arrange(DOWN, buff=0)
        
        # Add labels with better positioning
        row_labels = VGroup()
        col_labels = VGroup()
        
        for i, word in enumerate(words):
            row_label = Text(word, font_size=18, weight=BOLD)
            col_label = Text(word, font_size=18, weight=BOLD)
            row_labels.add(row_label)
            col_labels.add(col_label)
        
        row_labels.arrange(DOWN, buff=0.8).next_to(squares, LEFT, buff=0.5)
        col_labels.arrange(RIGHT, buff=0.8).next_to(squares, UP, buff=0.5)
        
        # Add axis labels
        from_label = Text("From →", font_size=16, color=BLUE).next_to(col_labels, UP, buff=0.3)
        to_label = Text("To ↓", font_size=16, color=BLUE).next_to(row_labels, LEFT, buff=0.3)
        
        matrix_group = VGroup(squares, row_labels, col_labels, from_label, to_label)
        matrix_group.center()
        
        self.play(FadeIn(matrix_group))
        self.wait(2)
        
        # Highlight some interesting connections
        self.highlight_attention_patterns(squares, words)
        
    def highlight_attention_patterns(self, squares, words):
        # Highlight cat -> sat connection (strongest relationship)
        cat_sat_square = squares[1][2]  # cat attending to sat (0.6)
        highlight1 = SurroundingRectangle(cat_sat_square, color=YELLOW, buff=0.1)
        
        explanation1 = Text("'cat' strongly attends to 'sat' (0.60)", font_size=16, color=YELLOW)
        explanation1.next_to(squares, DOWN, buff=0.5)
        
        self.play(Create(highlight1), Write(explanation1))
        self.wait(2)
        
        # Highlight on -> mat connection
        on_mat_square = squares[3][4]  # on attending to mat (0.65)
        highlight2 = SurroundingRectangle(on_mat_square, color=GREEN, buff=0.1)
        
        explanation2 = Text("'on' strongly attends to 'mat' (0.65)", font_size=16, color=GREEN)
        explanation2.next_to(explanation1, DOWN, buff=0.2)
        
        self.play(Create(highlight2), Write(explanation2))
        self.wait(2)
        
        # Highlight The -> mat connection  
        the_mat_square = squares[0][4]  # The attending to mat (0.6)
        highlight3 = SurroundingRectangle(the_mat_square, color=BLUE, buff=0.1)
        
        explanation3 = Text("'The' attends to 'mat' (determiner-noun)", font_size=16, color=BLUE)
        explanation3.next_to(explanation2, DOWN, buff=0.2)
        
        self.play(Create(highlight3), Write(explanation3))
        self.wait(2)
        
        self.play(FadeOut(highlight1), FadeOut(highlight2), FadeOut(highlight3))
        self.wait(1)

class MultiHeadAttentionScene(Scene):
    def construct(self):
        title = Text("Multi-Head Attention", font_size=36, color=BLUE)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        # Show multiple attention heads as matrices
        head_colors = [RED, GREEN, BLUE, ORANGE, PURPLE, YELLOW, PINK, TEAL]
        num_heads = 8
        
        # Input representation
        input_text = Text("Input Sequence", font_size=20)
        input_boxes = []
        for i in range(5):
            box = Rectangle(width=1, height=0.5, color=WHITE, fill_opacity=0.3)
            input_boxes.append(box)
        
        input_row = VGroup(*input_boxes).arrange(RIGHT, buff=0.1)
        input_group = VGroup(input_text, input_row).arrange(DOWN, buff=0.3).shift(UP * 2.5)
        
        self.play(FadeIn(input_group))
        self.wait(1)
        
        # Create attention heads as matrices (not vectors)
        head_groups = []
        for h in range(num_heads):
            head_label = Text(f"Head {h+1}", font_size=12, color=head_colors[h])
            
            # Create attention matrix for each head (5x5)
            head_matrix = []
            for i in range(5):
                matrix_row = []
                for j in range(5):
                    cell = Rectangle(width=0.15, height=0.15, color=head_colors[h], 
                                   fill_opacity=0.7, stroke_width=0.5)
                    matrix_row.append(cell)
                row_group = VGroup(*matrix_row).arrange(RIGHT, buff=0.01)
                head_matrix.append(row_group)
            
            matrix_group = VGroup(*head_matrix).arrange(DOWN, buff=0.01)
            head_group = VGroup(head_label, matrix_group).arrange(DOWN, buff=0.1)
            head_groups.append(head_group)
        
        # Arrange heads in 2 rows of 4
        top_heads = VGroup(*head_groups[:4]).arrange(RIGHT, buff=0.4)
        bottom_heads = VGroup(*head_groups[4:]).arrange(RIGHT, buff=0.4)
        all_heads = VGroup(top_heads, bottom_heads).arrange(DOWN, buff=0.8)
        all_heads.center().shift(DOWN * 0.2)
        
        # Show heads appearing one by one
        for head_group in head_groups:
            self.play(FadeIn(head_group))
            self.wait(0.2)
        
        self.wait(1)
        
        # Show matrix multiplication step
        self.show_matrix_multiplication(head_groups, head_colors)
        
        # Show concatenation
        self.show_concatenation(head_colors)
        
    def show_matrix_multiplication(self, head_groups, head_colors):
        # Clear and show matrix multiplication
        self.clear()
        
        title = Text("Matrix Multiplication in Each Head", font_size=28, color=BLUE)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        # Show one head's computation in detail
        head_color = head_colors[0]
        
        # Q, K, V matrices for one head
        q_label = Text("Q", font_size=24, color=head_color)
        k_label = Text("K", font_size=24, color=head_color)  
        v_label = Text("V", font_size=24, color=head_color)
        
        # Create matrix representations
        q_matrix = self.create_matrix(3, 5, head_color, 0.4)
        k_matrix = self.create_matrix(3, 5, head_color, 0.4)
        v_matrix = self.create_matrix(3, 5, head_color, 0.4)
        
        q_group = VGroup(q_label, q_matrix).arrange(DOWN, buff=0.2)
        k_group = VGroup(k_label, k_matrix).arrange(DOWN, buff=0.2)
        v_group = VGroup(v_label, v_matrix).arrange(DOWN, buff=0.2)
        
        qkv_groups = VGroup(q_group, k_group, v_group).arrange(RIGHT, buff=1).shift(UP * 1)
        
        self.play(FadeIn(qkv_groups))
        self.wait(1)
        
        # Show multiplication: Q @ K^T
        mult_symbol1 = Text("×", font_size=32, color=WHITE)
        mult_symbol1.move_to((q_group.get_right() + k_group.get_left()) / 2)
        
        self.play(Write(mult_symbol1))
        
        # Result matrix (attention scores)
        result_matrix = self.create_matrix(5, 5, YELLOW, 0.6)
        result_label = Text("Attention Scores", font_size=18, color=YELLOW)
        result_group = VGroup(result_label, result_matrix).arrange(DOWN, buff=0.2)
        result_group.move_to(DOWN * 1.5)
        
        self.play(FadeIn(result_group))
        self.wait(2)
        
    def create_matrix(self, rows, cols, color, opacity):
        matrix_elements = []
        for i in range(rows):
            row_elements = []
            for j in range(cols):
                element = Rectangle(width=0.25, height=0.25, color=color, 
                                  fill_opacity=opacity, stroke_width=1)
                row_elements.append(element)
            row_group = VGroup(*row_elements).arrange(RIGHT, buff=0.02)
            matrix_elements.append(row_group)
        
        return VGroup(*matrix_elements).arrange(DOWN, buff=0.02)
        
    def show_concatenation(self, head_colors):
        # Clear and show concatenation
        self.clear()
        
        title = Text("Concatenation of All Heads", font_size=28, color=BLUE)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        # Show individual head outputs
        head_outputs = []
        for i, color in enumerate(head_colors):
            output_matrix = self.create_matrix(2, 5, color, 0.7)
            label = Text(f"H{i+1}", font_size=12, color=color)
            head_output = VGroup(label, output_matrix).arrange(DOWN, buff=0.1)
            head_outputs.append(head_output)
        
        # Arrange in two rows
        top_outputs = VGroup(*head_outputs[:4]).arrange(RIGHT, buff=0.3)
        bottom_outputs = VGroup(*head_outputs[4:]).arrange(RIGHT, buff=0.3)
        all_outputs = VGroup(top_outputs, bottom_outputs).arrange(DOWN, buff=0.5)
        all_outputs.center().shift(UP * 0.5)
        
        for output in head_outputs:
            self.play(FadeIn(output))
            self.wait(0.1)
        
        # Show concatenation arrow
        concat_arrow = Arrow(
            start=all_outputs.get_bottom() + DOWN * 0.2,
            end=DOWN * 2,
            color=WHITE,
            buff=0.1
        )
        concat_label = Text("Concatenate", font_size=18, color=WHITE)
        concat_label.next_to(concat_arrow, RIGHT, buff=0.2)
        
        self.play(GrowArrow(concat_arrow), Write(concat_label))
        
        # Show final concatenated matrix
        final_matrix = self.create_matrix(16, 5, WHITE, 0.5)  # 8 heads * 2 dims each = 16
        final_label = Text("Concatenated Output", font_size=18, color=WHITE)
        final_group = VGroup(final_label, final_matrix).arrange(DOWN, buff=0.3)
        final_group.move_to(DOWN * 2.5)
        
        self.play(FadeIn(final_group))
        self.wait(2)

class PositionalEncodingScene(Scene):
    def construct(self):
        title = Text("Positional Encoding", font_size=36, color=BLUE)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        # Explain the problem
        problem_text = Text("Problem: How to encode position without recurrence?", font_size=24, color=RED)
        problem_text.shift(UP * 1.5)
        self.play(Write(problem_text))
        self.wait(2)
        
        # Show the solution
        solution_text = Text("Solution: Sinusoidal Position Encoding", font_size=24, color=GREEN)
        solution_text.shift(UP * 0.5)
        self.play(Write(solution_text))
        self.wait(1)
        
        # Show the formulas
        formula1 = MathTex(r"PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)", font_size=24)
        formula2 = MathTex(r"PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)", font_size=24)
        
        formulas = VGroup(formula1, formula2).arrange(DOWN, buff=0.3).shift(DOWN * 0.5)
        
        self.play(Write(formula1))
        self.wait(1)
        self.play(Write(formula2))
        self.wait(2)
        
        # Create visual representation
        self.create_positional_encoding_visual()
        
    def create_positional_encoding_visual(self):
        # Clear and show visual
        self.clear()
        
        title = Text("Positional Encoding Visualization", font_size=32, color=BLUE)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        # Create accurate positional encoding computation
        seq_length = 10
        embed_dim = 8
        
        # Generate positional encodings using the actual formula
        pos_encoding = np.zeros((seq_length, embed_dim))
        for pos in range(seq_length):
            for i in range(embed_dim // 2):
                # Actual positional encoding formulas
                pos_encoding[pos, 2*i] = np.sin(pos / (10000 ** (2*i / embed_dim)))
                pos_encoding[pos, 2*i + 1] = np.cos(pos / (10000 ** (2*i / embed_dim)))
        
        # Normalize for better visualization (keep original values for display)
        pos_encoding_display = pos_encoding.copy()
        pos_encoding_norm = (pos_encoding + 1) / 2  # Scale to [0, 1] for colors
        
        # Create visual grid with actual values
        squares = VGroup()
        for i in range(seq_length):
            row = VGroup()
            for j in range(embed_dim):
                intensity = pos_encoding_norm[i, j]
                actual_value = pos_encoding_display[i, j]
                
                color = interpolate_color(BLUE, RED, intensity)
                square = Square(side_length=0.5, color=color, fill_opacity=0.8, stroke_color=BLACK, stroke_width=1)
                
                # Show actual computed values
                if abs(actual_value) < 0.01:
                    value_text = Text("0.00", font_size=8, color=WHITE if intensity > 0.5 else BLACK)
                else:
                    value_text = Text(f"{actual_value:.2f}", font_size=8, color=WHITE if intensity > 0.5 else BLACK)
                
                square_group = VGroup(square, value_text)
                row.add(square_group)
            row.arrange(RIGHT, buff=0.02)
            squares.add(row)
        
        squares.arrange(DOWN, buff=0.02)
        squares.center()
        
        # Add comprehensive labels
        pos_labels = VGroup()
        dim_labels = VGroup()
        
        for i in range(seq_length):
            label = Text(f"pos {i}", font_size=12, weight=BOLD)
            pos_labels.add(label)
        
        for j in range(embed_dim):
            if j % 2 == 0:
                label = Text(f"sin({j//2})", font_size=10, color=BLUE)
            else:
                label = Text(f"cos({j//2})", font_size=10, color=RED)
            dim_labels.add(label)
        
        pos_labels.arrange(DOWN, buff=0.52).next_to(squares, LEFT, buff=0.3)
        dim_labels.arrange(RIGHT, buff=0.52).next_to(squares, UP, buff=0.3)
        
        # Labels for axes with better descriptions
        pos_axis_label = Text("Sequence Position", font_size=16, color=GREEN).next_to(pos_labels, LEFT, buff=0.3)
        dim_axis_label = Text("Embedding Dimension (sin/cos pairs)", font_size=16, color=GREEN).next_to(dim_labels, UP, buff=0.3)
        
        # Add legend
        legend_title = Text("Color Legend:", font_size=14, weight=BOLD).to_edge(RIGHT, buff=1).shift(UP * 2)
        legend_low = Rectangle(width=0.3, height=0.3, color=BLUE, fill_opacity=0.8)
        legend_high = Rectangle(width=0.3, height=0.3, color=RED, fill_opacity=0.8)
        legend_low_text = Text("-1.0", font_size=12).next_to(legend_low, DOWN, buff=0.1)
        legend_high_text = Text("+1.0", font_size=12).next_to(legend_high, DOWN, buff=0.1)
        
        legend_group = VGroup(
            legend_title,
            VGroup(legend_low, legend_low_text).arrange(DOWN, buff=0.1),
            VGroup(legend_high, legend_high_text).arrange(DOWN, buff=0.1)
        ).arrange(DOWN, buff=0.3).to_edge(RIGHT, buff=1)
        
        visualization = VGroup(squares, pos_labels, dim_labels, pos_axis_label, dim_axis_label)
        
        self.play(FadeIn(visualization))
        self.play(FadeIn(legend_group))
        self.wait(2)
        
        # Add detailed explanation
        explanation1 = Text("Each position gets a unique sinusoidal pattern", font_size=18, color=WHITE)
        explanation2 = Text("Different frequencies for each dimension", font_size=16, color=GRAY)
        explanation_group = VGroup(explanation1, explanation2).arrange(DOWN, buff=0.2)
        explanation_group.next_to(squares, DOWN, buff=0.8)
        
        self.play(Write(explanation1))
        self.wait(1)
        self.play(Write(explanation2))
        self.wait(3)

class CompleteTransformerArchitectureScene(Scene):
    def construct(self):
        title = Text("Complete Transformer Architecture", font_size=32, color=BLUE)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        # Create encoder-decoder structure
        encoder = self.create_encoder_block()
        decoder = self.create_decoder_block()
        
        # Position encoder and decoder
        encoder.shift(LEFT * 3)
        decoder.shift(RIGHT * 3)
        
        self.play(FadeIn(encoder), FadeIn(decoder))
        self.wait(1)
        
        # Add data flow arrows
        self.add_data_flow(encoder, decoder)
        
    def create_encoder_block(self):
        # Encoder components
        input_embed = Rectangle(width=2, height=0.5, color=BLUE, fill_opacity=0.3)
        input_text = Text("Input\nEmbedding", font_size=12)
        input_group = VGroup(input_embed, input_text)
        
        pos_enc = Rectangle(width=2, height=0.5, color=ORANGE, fill_opacity=0.3)
        pos_text = Text("Positional\nEncoding", font_size=12)
        pos_group = VGroup(pos_enc, pos_text)
        
        self_att = Rectangle(width=2, height=0.8, color=RED, fill_opacity=0.3)
        att_text = Text("Multi-Head\nSelf-Attention", font_size=12)
        att_group = VGroup(self_att, att_text)
        
        ffn = Rectangle(width=2, height=0.8, color=GREEN, fill_opacity=0.3)
        ffn_text = Text("Feed-Forward\nNetwork", font_size=12)
        ffn_group = VGroup(ffn, ffn_text)
        
        encoder_stack = VGroup(input_group, pos_group, att_group, ffn_group).arrange(UP, buff=0.2)
        
        # Add "Encoder" label
        encoder_label = Text("ENCODER", font_size=16, color=BLUE)
        encoder_label.next_to(encoder_stack, UP, buff=0.3)
        
        return VGroup(encoder_label, encoder_stack)
        
    def create_decoder_block(self):
        # Decoder components
        output_embed = Rectangle(width=2, height=0.5, color=PURPLE, fill_opacity=0.3)
        output_text = Text("Output\nEmbedding", font_size=12)
        output_group = VGroup(output_embed, output_text)
        
        masked_att = Rectangle(width=2, height=0.8, color=YELLOW, fill_opacity=0.3)
        masked_text = Text("Masked\nSelf-Attention", font_size=12)
        masked_group = VGroup(masked_att, masked_text)
        
        cross_att = Rectangle(width=2, height=0.8, color=PINK, fill_opacity=0.3)
        cross_text = Text("Cross\nAttention", font_size=12)
        cross_group = VGroup(cross_att, cross_text)
        
        ffn_dec = Rectangle(width=2, height=0.8, color=GREEN, fill_opacity=0.3)
        ffn_dec_text = Text("Feed-Forward\nNetwork", font_size=12)
        ffn_dec_group = VGroup(ffn_dec, ffn_dec_text)
        
        decoder_stack = VGroup(output_group, masked_group, cross_group, ffn_dec_group).arrange(UP, buff=0.2)
        
        # Add "Decoder" label
        decoder_label = Text("DECODER", font_size=16, color=PURPLE)
        decoder_label.next_to(decoder_stack, UP, buff=0.3)
        
        return VGroup(decoder_label, decoder_stack)
        
    def add_data_flow(self, encoder, decoder):
        # Arrow from encoder to decoder (cross-attention)
        cross_arrow = Arrow(
            start=encoder.get_right() + RIGHT * 0.1,
            end=decoder.get_left() + LEFT * 0.1,
            color=PINK,
            buff=0.1
        )
        
        cross_label = Text("K, V", font_size=12, color=PINK)
        cross_label.next_to(cross_arrow, UP, buff=0.1)
        
        self.play(GrowArrow(cross_arrow), Write(cross_label))
        self.wait(2)

if __name__ == "__main__":
    import sys
    from manim import config
    
    # Configuration for video quality
    config.quality = "medium_quality"  # You can change to "high_quality" for better output
    config.preview = False  # Set to True if you want to preview
    
    # List of all scenes to render with their intended filenames
    scenes = [
        (TransformerOverviewScene, "TransformerOverview"),
        (MultiHeadAttentionScene, "MultiHeadAttention"), 
        (PositionalEncodingScene, "PositionalEncoding"),
        (CompleteTransformerArchitectureScene, "CompleteTransformerArchitecture")
    ]
    
    print("Starting video generation...")
    
    for i, (scene_class, filename) in enumerate(scenes, 1):
        print(f"Rendering scene {i}/{len(scenes)}: {scene_class.__name__}")
        try:
            # Set unique output filename for each scene
            config.output_file = f"{filename}.mp4"
            
            # Create and render the scene
            scene = scene_class()
            scene.render()
            print(f"✓ Successfully rendered {scene_class.__name__} as {filename}.mp4")
        except Exception as e:
            print(f"✗ Error rendering {scene_class.__name__}: {e}")
    
    print("Video generation complete!")
    print("Videos saved to: media/videos/720p30/")
