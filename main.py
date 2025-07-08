
from manim import *

# class TokenizationScene(Scene):
#     def construct(self):
#         # Create the sentence
#         sentence = Text("I love machine learning", font_size=36)
#         self.play(Write(sentence))
#         self.wait(1)
        
#         # Split into tokens
#         tokens = ["I", "love", "machine", "learning"]
#         token_mobjects = []
        
#         for i, token in enumerate(tokens):
#             token_text = Text(token, font_size=28).set_color(PURPLE)
#             token_box = SurroundingRectangle(token_text, buff=0.2, color=PURPLE)
#             token_group = VGroup(token_box, token_text)
#             token_mobjects.append(token_group)
        
#         token_row = VGroup(*token_mobjects).arrange(RIGHT, buff=0.5)
        
#         self.play(
#             FadeOut(sentence),
#             FadeIn(token_row)
#         )
#         self.wait(1)

# class EmbeddingScene(Scene):
#     def construct(self):
#         # Create tokens
#         tokens = ["I", "love", "machine", "learning"]
#         token_mobjects = []
        
#         for token in tokens:
#             token_text = Text(token, font_size=28).set_color(PURPLE)
#             token_box = SurroundingRectangle(token_text, buff=0.2, color=PURPLE)
#             token_group = VGroup(token_box, token_text)
#             token_mobjects.append(token_group)
        
#         token_row = VGroup(*token_mobjects).arrange(RIGHT, buff=0.5).to_edge(UP)
        
#         # Create embedding vectors
#         embedding_vectors = []
#         for i in range(4):
#             # Create a random-looking vector
#             vector_values = np.random.rand(6) * 2 - 1
#             vector_text = MathTex(
#                 r"\begin{bmatrix} " + 
#                 r" \\ ".join([f"{value:.2f}" for value in vector_values]) + 
#                 r" \end{bmatrix}",
#                 font_size=24
#             )
#             embedding_vectors.append(vector_text)
        
#         vector_row = VGroup(*embedding_vectors).arrange(RIGHT, buff=1.0).to_edge(DOWN)
        
#         # Show the process
#         self.play(FadeIn(token_row))
#         self.wait(1)
        
#         self.play(FadeIn(vector_row))
        
#         # Draw arrows connecting tokens to vectors
#         arrows = []
#         for i in range(4):
#             arrow = Arrow(
#                 start=token_mobjects[i].get_bottom() + DOWN * 0.2,
#                 end=embedding_vectors[i].get_top() + UP * 0.2,
#                 color=BLUE
#             )
#             arrows.append(arrow)
        
#         self.play(*[GrowArrow(arrow) for arrow in arrows])
#         self.wait(1)

# class MultiHeadAttentionScene(Scene):
#     def construct(self):
#         # Create input representation
#         input_text = Text("Input Representation", font_size=36)
#         input_box = SurroundingRectangle(input_text, buff=0.3, color=GREEN)
#         input_group = VGroup(input_box, input_text).to_edge(UP)
        
#         # Create attention heads
#         heads = []
#         for i in range(8):
#             head_text = Text(f"Head {i+1}", font_size=24)
#             head_circle = Circle(radius=0.5, color=BLUE).set_fill(BLUE, opacity=0.3)
#             head_group = VGroup(head_circle, head_text)
#             heads.append(head_group)
        
#         head_row = VGroup(*heads[:4]).arrange(RIGHT, buff=0.5)
#         head_row2 = VGroup(*heads[4:]).arrange(RIGHT, buff=0.5)
#         all_heads = VGroup(head_row, head_row2).arrange(DOWN, buff=0.5).center()
        
#         # Create output representation
#         output_text = Text("Multi-Head Attention Output", font_size=30)
#         output_box = SurroundingRectangle(output_text, buff=0.3, color=RED)
#         output_group = VGroup(output_box, output_text).to_edge(DOWN)
        
#         # Show the process
#         self.play(FadeIn(input_group))
#         self.wait(1)
        
#         self.play(FadeIn(all_heads))
        
#         # Show arrows from input to heads
#         input_arrows = []
#         for head in heads:
#             arrow = Arrow(
#                 start=input_group.get_bottom() + DOWN * 0.1,
#                 end=head.get_top() + UP * 0.1,
#                 color=YELLOW
#             )
#             input_arrows.append(arrow)
        
#         self.play(*[GrowArrow(arrow) for arrow in input_arrows])
#         self.wait(1)
        
#         # Show arrows from heads to output
#         output_arrows = []
#         for head in heads:
#             arrow = Arrow(
#                 start=head.get_bottom() + DOWN * 0.1,
#                 end=output_group.get_top() + UP * 0.1,
#                 color=YELLOW
#             )
#             output_arrows.append(arrow)
        
#         self.play(FadeIn(output_group))
#         self.play(*[GrowArrow(arrow) for arrow in output_arrows])
#         self.wait(1)

class QKVComputationScene(Scene):
    def construct(self):
        # Create input vector
        input_label = Text("Input Vector", font_size=24).to_edge(UP)
        input_vector = MathTex(r"\begin{bmatrix} 0.2 \\ 0.5 \\ -0.3 \\ 0.7 \end{bmatrix}", font_size=30)
        input_group = VGroup(input_label, input_vector).arrange(DOWN).to_edge(LEFT)
        
        # Create QKV projections
        projection_labels = [
            Text("Query (Q)", font_size=24, color=BLUE),
            Text("Key (K)", font_size=24, color=GREEN),
            Text("Value (V)", font_size=24, color=RED)
        ]
        
        projection_matrices = [
            MathTex(r"W_Q = \begin{bmatrix} 0.1 & 0.3 & -0.2 & 0.4 \\ 0.5 & 0.1 & 0.3 & -0.1 \end{bmatrix}", font_size=24),
            MathTex(r"W_K = \begin{bmatrix} 0.2 & -0.1 & 0.4 & 0.3 \\ 0.1 & 0.5 & 0.2 & -0.3 \end{bmatrix}", font_size=24),
            MathTex(r"W_V = \begin{bmatrix} 0.4 & 0.2 & 0.1 & -0.3 \\ -0.2 & 0.4 & 0.3 & 0.1 \end{bmatrix}", font_size=24)
        ]
        
        result_vectors = [
            MathTex(r"Q = \begin{bmatrix} 0.32 \\ 0.24 \end{bmatrix}", font_size=30),
            MathTex(r"K = \begin{bmatrix} 0.25 \\ 0.18 \end{bmatrix}", font_size=30),
            MathTex(r"V = \begin{bmatrix} 0.09 \\ 0.42 \end{bmatrix}", font_size=30)
        ]
        
        qkv_groups = []
        for i in range(3):
            group = VGroup(
                projection_labels[i],
                projection_matrices[i],
                result_vectors[i]
            ).arrange(DOWN, buff=0.3)
            qkv_groups.append(group)
        
        VGroup(*qkv_groups).arrange(RIGHT, buff=1.0).center()
        
        # Show the process
        self.play(FadeIn(input_group))
        self.wait(1)
        
        for i, group in enumerate(qkv_groups):
            self.play(FadeIn(group[0]))  # Show label
            self.play(FadeIn(group[1]))  # Show matrix
            
            # Show multiplication arrow
            arrow = Arrow(
                start=input_vector.get_right() + RIGHT * 0.2,
                end=group[1].get_left() + LEFT * 0.2,
                color=YELLOW
            )
            
            self.play(GrowArrow(arrow))
            self.play(FadeIn(group[2]))  # Show result
            self.wait(0.5)
        
        self.wait(1)

class CompleteTransformerScene(Scene):
    def construct(self):
        # Create title
        title = Text("Transformer: Encoder-Decoder Architecture", font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))
        
        # Create encoder and decoder blocks
        encoder_label = Text("Encoder", font_size=30, color=GREEN)
        encoder_box = Rectangle(height=4, width=3, color=GREEN).set_fill(GREEN, opacity=0.1)
        encoder_group = VGroup(encoder_box, encoder_label)
        
        decoder_label = Text("Decoder", font_size=30, color=RED)
        decoder_box = Rectangle(height=4, width=3, color=RED).set_fill(RED, opacity=0.1)
        decoder_group = VGroup(decoder_box, decoder_label)
        
        VGroup(encoder_group, decoder_group).arrange(RIGHT, buff=2.0).center()
        
        self.play(FadeIn(encoder_group), FadeIn(decoder_group))
        
        # Create input and output
        input_text = Text("I love machine learning", font_size=24)
        input_box = SurroundingRectangle(input_text, buff=0.2, color=BLUE)
        input_group = VGroup(input_box, input_text).next_to(encoder_group, UP)
        
        output_text = Text("J'aime l'apprentissage automatique", font_size=24)
        output_box = SurroundingRectangle(output_text, buff=0.2, color=YELLOW)
        output_group = VGroup(output_box, output_text).next_to(decoder_group, DOWN)
        
        # Show the process
        self.play(FadeIn(input_group))
        
        # Arrow from input to encoder
        input_arrow = Arrow(
            start=input_group.get_bottom() + DOWN * 0.1,
            end=encoder_group.get_top() + UP * 0.1,
            color=BLUE
        )
        self.play(GrowArrow(input_arrow))
        
        # Processing in encoder (pulsing effect)
        self.play(
            encoder_box.animate.set_fill(GREEN, opacity=0.3),
            rate_func=there_and_back,
            run_time=2
        )
        
        # Arrow from encoder to decoder
        encoder_decoder_arrow = Arrow(
            start=encoder_group.get_right() + RIGHT * 0.1,
            end=decoder_group.get_left() + LEFT * 0.1,
            color=PURPLE
        )
        self.play(GrowArrow(encoder_decoder_arrow))
        
        # Processing in decoder (pulsing effect)
        self.play(
            decoder_box.animate.set_fill(RED, opacity=0.3),
            rate_func=there_and_back,
            run_time=2
        )
        
        # Output from decoder
        self.play(FadeIn(output_group))
        output_arrow = Arrow(
            start=decoder_group.get_bottom() + DOWN * 0.1,
            end=output_group.get_top() + UP * 0.1,
            color=YELLOW
        )
        self.play(GrowArrow(output_arrow))
        
        self.wait(2)